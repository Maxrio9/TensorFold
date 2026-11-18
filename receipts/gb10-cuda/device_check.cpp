#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <vector>
static void ok(CUresult r) {
    if (r == CUDA_SUCCESS) return;
    const char *why = nullptr; cuGetErrorString(r, &why);
    throw std::runtime_error(why ? why : "CUDA error");
}
struct Buffer {
    CUdeviceptr p{};
    explicit Buffer(size_t bytes) { ok(cuMemAlloc(&p, bytes)); }
    ~Buffer() { cuMemFree(p); }
    Buffer(const Buffer&) = delete;
};
static CUfunction entry(CUmodule m, const char *name) {
    CUfunction f; ok(cuModuleGetFunction(&f, m, name)); return f;
}
static uint16_t bf(float x) {
    uint32_t v; std::memcpy(&v, &x, 4); return uint16_t((v + 0x7fff + ((v>>16)&1)) >> 16);
}
static void attention(CUmodule mods[2], bool paged, int p0, int W, int page) {
    const int H=24, HK=4, D=256, G=H/HK, n=p0+W;
    int pages=(n+page-1)/page, cap=paged?page:n;
    size_t nr=HK*cap, nb=nr*D, stride=2*nb+2*nr*4;
    size_t outbytes=size_t(W)*H*D*2;
    Buffer kv(stride*(paged?pages:1)), query(outbytes), pt(pages*4);
    Buffer out0(outbytes+128), out1(outbytes+128);
    std::vector<unsigned char> host(stride*(paged?pages:1));
    std::mt19937 rng(20261117);
    for (int p=0; p<(paged?pages:1); ++p) {
        auto base=host.data()+p*stride;
        for(size_t i=0;i<2*nb;++i) base[i]=(rng()&0x80) | (rng()%73);
        for(size_t i=0;i<2*nr;++i) {
            float v=float(1+rng()%31)/16; std::memcpy(base+2*nb+4*i,&v,4);
        }
    }
    std::vector<uint16_t> q(outbytes/2);
    for (auto &x:q) x=bf(float(int(rng()%127)-63)/48);
    std::vector<int> ids(pages);
    for(int p=0;p<pages;++p) ids[p]=pages-1-p;
    ok(cuMemcpyHtoD(kv.p,host.data(),host.size())); ok(cuMemcpyHtoD(query.p,q.data(),outbytes));
    ok(cuMemcpyHtoD(pt.p,ids.data(),pages*4));
    const char *sym=paged?"_ZN20tf_prefill_attention14hpattn8_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifiPKixi":"_ZN20tf_prefill_attention20hpattn8_whole_kernelILi256ELi8ELi2ELi2EEEvPK13__nv_bfloat16PKhS5_PKfS7_PS1_iiiiifi";
    CUfunction f[2]={entry(mods[0],sym),entry(mods[1],sym)};
    CUdeviceptr k=kv.p, v=k+nb, ks=v+nb, vs=ks+nr*4, dst[]={out0.p+64,out1.p+64};
    int h=H,hk=HK,g=G, full=n; float scale=1.f/16;
    void *args[]={&query.p,&k,&v,&ks,&vs,nullptr,&p0,&W,&h,&hk,&g,&scale,&full,&pt.p,&stride,&page};
    auto launch=[&](int i) {
        args[5]=&dst[i]; ok(cuLaunchKernel(f[i],(W+63)/64,HK*(G/2),1,256,1,1,98304,nullptr,args,nullptr));
    };
    std::vector<unsigned char> results[2];
    for(int i=0;i<2;++i) {
        ok(cuFuncSetAttribute(f[i],CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,98304));
        results[i].resize(outbytes+128);
        ok(cuMemsetD8(dst[i]-64,0xa5,outbytes+128));
        ok(cuMemsetD8(dst[i],i?0xff:0x44,outbytes)); launch(i);
        ok(cuCtxSynchronize());
        ok(cuMemcpyDtoH(results[i].data(),dst[i]-64,outbytes+128));
        for(int j=0;j<64;++j)
            if(results[i][j]!=0xa5 || results[i][64+outbytes+j]!=0xa5) throw std::runtime_error("output guard");
    }
    if(results[0]!=results[1]) throw std::runtime_error("output mismatch");
    CUevent a,b; ok(cuEventCreate(&a,0)); ok(cuEventCreate(&b,0));
    std::vector<float> samples[2];
    for(int r=0;r<10;++r) for(int j=0;j<2;++j) {
        int i=(r+j)%2; ok(cuEventRecord(a,nullptr)); for(int rep=0;rep<5;++rep) launch(i);
        ok(cuEventRecord(b,nullptr)); ok(cuEventSynchronize(b)); float ms; ok(cuEventElapsedTime(&ms,a,b)); samples[i].push_back(ms/5);
    }
    ok(cuEventDestroy(a)); ok(cuEventDestroy(b));
    for(auto &s:samples) std::sort(s.begin(),s.end());
    printf("fp8 paged=%d prefix=%d rows=%d page=%d parity=pass base_ms=%.5f patch_ms=%.5f\n",paged,p0,W,page,(samples[0][4]+samples[0][5])/2,(samples[1][4]+samples[1][5])/2);
}
int main(int argc,char** argv) {
    if(argc!=4) return 2;
    ok(cuInit(0)); CUdevice dev; ok(cuDeviceGet(&dev,0)); CUcontext ctx;
    ok(cuDevicePrimaryCtxRetain(&ctx,dev)); ok(cuCtxSetCurrent(ctx));
    size_t free,total; ok(cuMemGetInfo(&free,&total));
    if(free < (size_t(std::atoi(argv[3]))<<30)+(128ULL<<20)) throw std::runtime_error("GPU memory reserve");
    CUmodule m[2]; ok(cuModuleLoad(&m[0],argv[1])); ok(cuModuleLoad(&m[1],argv[2]));
    for(bool p:{false,true}) for(auto s:{std::array<int,3>{0,1,48},{31,33,48},{255,65,256},{1000,193,48},{3073,128,256}}) attention(m,p,s[0],s[1],s[2]);
    ok(cuModuleUnload(m[0])); ok(cuModuleUnload(m[1])); ok(cuDevicePrimaryCtxRelease(dev));
}
