// PROTOTYPE, beside proto_vs_shared.py: what the GPU compiler makes of a vertex function in each form. For every library given, a render pipeline of its entry point and a
// fragment function that reads nothing is added to one binary archive (the GPU compiler runs then, as for the host's pipeline archive); the archive's size and the time say how the
// forms compare. A pulling entry point needs no vertex descriptor, so it serves for any shader.
// Build: clang -fobjc-arc -O1 -framework Metal -framework Foundation proto_vs_gpu.m -o <work>/proto_vs_gpu
// Usage: proto_vs_gpu <archive to write> <entry point> <library>...    prints: libraries added, failures, archive bytes, ms
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc < 4) { fprintf(stderr, "usage: %s <archive> <entry> <library>...\n", argv[0]); return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError *e = nil;
        id<MTLLibrary> fl = [dev newLibraryWithSource:@"#include <metal_stdlib>\nusing namespace metal;\nfragment float4 f0() { return float4(1.0); }\n" options:nil error:&e];
        if (!fl) { fprintf(stderr, "fragment: %s\n", e.localizedDescription.UTF8String); return 1; }
        id<MTLFunction> ff = [fl newFunctionWithName:@"f0"];
        id<MTLBinaryArchive> ar = [dev newBinaryArchiveWithDescriptor:[MTLBinaryArchiveDescriptor new] error:&e];
        if (!ar) { fprintf(stderr, "archive: %s\n", e.localizedDescription.UTF8String); return 1; }
        NSString *entry = @(argv[2]); int added = 0, failed = 0; mach_timebase_info_data_t tb; mach_timebase_info(&tb); uint64_t ns = 0;
        for (int i = 3; i < argc; i++) {
            id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(argv[i])] error:&e];
            id<MTLFunction> vf = lib ? [lib newFunctionWithName:entry] : nil;
            if (vf && vf.functionConstantsDictionary.count) vf = [lib newFunctionWithName:entry constantValues:[MTLFunctionConstantValues new] error:&e];   // as the host specializes one
            if (!vf) { fprintf(stderr, "%s: no %s (%s)\n", argv[i], argv[2], e ? e.localizedDescription.UTF8String : "-"); failed++; continue; }
            MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new];
            d.vertexFunction = vf; d.fragmentFunction = ff; d.colorAttachments[0].pixelFormat = MTLPixelFormatRGBA8Unorm; d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float;
            MTLVertexDescriptor *vd = nil; int na = 0;   // a function with stage inputs: each one a 16-byte element of its own class, all from one buffer
            for (MTLVertexAttribute *a in vf.vertexAttributes) {
                if (!a.active) continue;
                if (!vd) vd = [MTLVertexDescriptor vertexDescriptor];
                MTLDataType t = a.attributeType;
                vd.attributes[a.attributeIndex].format = t == MTLDataTypeUInt4 ? MTLVertexFormatUInt4 : t == MTLDataTypeInt4 ? MTLVertexFormatInt4 : MTLVertexFormatFloat4;
                vd.attributes[a.attributeIndex].offset = 16 * na++; vd.attributes[a.attributeIndex].bufferIndex = 30; }
            if (vd) { vd.layouts[30].stride = 16 * na; d.vertexDescriptor = vd; }
            if (getenv("PV_ATTRS")) fprintf(stderr, "%s %s: %d stage inputs\n", argv[i], argv[2], na);   // a pulling entry point must have none
            uint64_t t0 = mach_absolute_time();
            BOOL ok = [ar addRenderPipelineFunctionsWithDescriptor:d error:&e];
            ns += (mach_absolute_time() - t0) * tb.numer / tb.denom;
            if (!ok) { fprintf(stderr, "%s: %s\n", argv[i], e.localizedDescription.UTF8String); failed++; continue; }
            added++;
        }
        NSURL *out = [NSURL fileURLWithPath:@(argv[1])];
        [[NSFileManager defaultManager] removeItemAtURL:out error:nil];
        if (![ar serializeToURL:out error:&e]) { fprintf(stderr, "serialize: %s\n", e.localizedDescription.UTF8String); return 1; }
        unsigned long long size = [[[NSFileManager defaultManager] attributesOfItemAtPath:out.path error:nil] fileSize];
        printf("%d\t%d\t%llu\t%.1f\n", added, failed, size, ns / 1e6);
    }
    return 0;
}
