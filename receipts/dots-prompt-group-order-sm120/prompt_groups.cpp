#include <cuda.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

void ck(CUresult rc) {
    if (!rc) return;
    const char *text = nullptr;
    cuGetErrorString(rc, &text);
    throw std::runtime_error(text ? text : "CUDA call failed");
}
struct Buffer {
    CUdeviceptr ptr;
    explicit Buffer(size_t bytes) { ck(cuMemAlloc(&ptr,bytes)); }
    ~Buffer() { cuMemFree(ptr); }
};
std::string product(int bm, int bn, int wm, int wide, int fused) {
    return "_ZN7tf_qmmf11qmmf_kernelILi4ELi"+std::to_string(bm)+"ELi"+std::to_string(bn)
        +"ELi"+std::to_string(wm)+"ELi4ELi"+std::to_string(bm==128?3:4)
        +"ELb"+std::to_string(wide)+"ELb0ELb"+std::to_string(fused)+"EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
}
void fill(Buffer &b, size_t words, std::mt19937 &rng) {
    std::vector<uint16_t> block(std::min<size_t>(1<<19,words));
    for (size_t i=0;i<words;i+=block.size()) {
        auto n=std::min(block.size(),words-i);
        for (size_t j=0;j<n;++j) {
            auto u=rng();
            block[j]=uint16_t((u&0x8000)|((117+(u>>16)%12)<<7)|(u&127));
        }
        ck(cuMemcpyHtoD(b.ptr+2*i,block.data(),2*n));
    }
}
void test(CUmodule mod,const char *name,int n,int k,int wide, int seed, int chosen) {
    int m=1024, npad=(n+127)/128*128, sk=1;
    for (int groups=k/64; sk<8 && ((n+63)/64)*sk<192 && groups%(2*sk)==0 && groups/(2*sk)>=8;) sk*=2;
    int fused=sk>1;
    std::array<CUfunction,2> f;
    auto old=product(64,64,1,wide,fused);
    auto newer=fused?product(128,128,2,wide,1):"_ZN7tf_qmmf22qmmf_prompt_k32_kernelILb"+std::to_string(wide)+"EEEvPK13__nv_bfloat16PKhS5_fPvPfiiiiiii";
    ck(cuModuleGetFunction(&f[0],mod,old.c_str())); ck(cuModuleGetFunction(&f[1],mod,newer.c_str()));
    int smem[]={65536, fused?98432:49152};
    for (int v=0;v<2;v++) ck(cuFuncSetAttribute(f[v],CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,smem[v]));
    size_t weights=size_t(npad)*k;
    int copies=int((192ULL*1024*1024 + weights*2-1)/(weights*2));
    size_t outsize=size_t(m)*n*(wide?4:2);
    Buffer w(weights*2*copies),x(size_t(m)*k*2),a(outsize+128),b(outsize+128);
    std::mt19937 rng(seed);
    fill(w,weights*copies,rng);fill(x,size_t(m)*k,rng);
    // Case 0 is the shipped tile/group; case 1 enables the prompt tile alone.
    // Cases 2-5 change group height independently, cases 6-9 combine both switches.
    auto launch=[&](int config,int which) {
        const int v=config==1 || config>=6;
        int bm=v?128:64,bn=v?128:64, band=std::max(1,std::min((m+bm-1)/bm,(12<<20)/(bm*k*2)));
        if(config>=2)band=std::min((m+bm-1)/bm,1<<((config-2)%4));
        CUdeviceptr wp=w.ptr+which*weights*2, y=(config?b.ptr:a.ptr)+64, empty=0;
        float scale=1;
        void *args[]={&x.ptr,&wp,&empty,&scale,&y,&empty,&m,&n,&k,&sk,&npad,&k,&band};
        ck(cuLaunchKernel(f[v],((m+bm-1)/bm)*((n+bn-1)/bn),1,1,v?256:128,1,1,smem[v],nullptr,args,nullptr));
    };
    std::vector<int> configs;
    if(chosen<0)configs={0,1,2,3,4,5,6,7,8,9};
    else configs={0,1,chosen};
    for (int config:configs) {
        if(!config)continue;
        for(int i=0;i<5;++i) {
            ck(cuMemsetD8(a.ptr,0xa5,outsize+128));ck(cuMemsetD8(b.ptr,0xa5,outsize+128));
            ck(cuMemsetD8(a.ptr+64,0x11,outsize));ck(cuMemsetD8(b.ptr+64,0x77,outsize));
            launch(0,i%copies);launch(config,i%copies);
            std::vector<unsigned char> first(outsize+128),second(outsize+128);
            ck(cuMemcpyDtoH(first.data(),a.ptr,outsize+128));ck(cuMemcpyDtoH(second.data(),b.ptr,outsize+128));
            if(first!=second)throw std::runtime_error(std::string("output mismatch ")+name);
            for(int j=0;j<64;j++)if(first[j]!=0xa5||first[outsize+64+j]!=0xa5)throw std::runtime_error("output guard");
        }
    }
    CUevent e1,e2;ck(cuEventCreate(&e1,0));ck(cuEventCreate(&e2,0));
    const int trials=chosen<0?10:12;
    for(int round=0;round<trials;++round) {
        for(size_t side=0;side<configs.size();++side) {
            const int v=configs[(side+round)%configs.size()];
            ck(cuEventRecord(e1,nullptr));
            for(int j=0;j<copies;++j)launch(v,(j+round)%copies);
            ck(cuEventRecord(e2,nullptr));ck(cuEventSynchronize(e2));
            float elapsed;ck(cuEventElapsedTime(&elapsed,e1,e2));
            printf("%s %d %d %d %.9f\n",name,seed,round,v,elapsed/copies);
        }
        fflush(stdout);
    }
    ck(cuEventDestroy(e1));ck(cuEventDestroy(e2));
}
int main(int argc,char **argv) {
    try{
        if(argc!=8)return 2;
        int seed=std::stoi(argv[3]);
        ck(cuInit(0));CUdevice dev;ck(cuDeviceGet(&dev,0));CUcontext ctx;ck(cuDevicePrimaryCtxRetain(&ctx,dev));ck(cuCtxSetCurrent(ctx));
        size_t free,total;ck(cuMemGetInfo(&free,&total));if(free<(size_t(std::stoul(argv[2]))<<30)+(1ULL<<30))throw std::runtime_error("insufficient memory under live reserve");
        CUmodule mod;ck(cuModuleLoad(&mod,argv[1]));
        test(mod,"qkv",2048,1536,1,seed,std::stoi(argv[4]));test(mod,"o",1536,1536,0,seed,std::stoi(argv[5]));test(mod,"gate_up",17920,1536,0,seed,std::stoi(argv[6]));test(mod,"down",1536,8960,0,seed,std::stoi(argv[7]));
        ck(cuModuleUnload(mod));ck(cuDevicePrimaryCtxRelease(dev));
    }catch(const std::exception &ex){fprintf(stderr,"%s\n",ex.what());return 1;}
}
