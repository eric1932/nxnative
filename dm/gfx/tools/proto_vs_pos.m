// PROTOTYPE, beside proto_vs_shared.py: what the vertex functions of one shader compute on this Mac's GPU, compared bit for bit. Every function named is given the same vertices
// (random but valid: float inputs in a range the seed picks, integer inputs 0-7), the same uniform data (float members random, integer members small, a block of only uint4 rows taken
// as the driver's sizes, 0x10000 each; the seed picks the ranges), the same textures, and draws the same random triangles.
//   raster   each function draws <triangles> triangles with depth test LESS into a Depth32Float target and, through the fragment functions given (one pass each, which copy up to
//            8 of the shader's float4 outputs to RGBA32Float targets), its outputs. The depth image and every colour image of a function are compared with the first function's:
//            a position that differs in any bit moves the depth of the pixels its triangle covers, as t_pull_inv.m relies on (there with depth EQUAL).
//   capture  each function is a geometry capture entry point (main0_gs): it writes every output word of every vertex to a buffer (dmsh_geometry.py's 144-word record), which is
//            compared with the first function's, word for word.
// A function with stage inputs takes them through a vertex descriptor (its shadow inputs at attributes 16-30, dmsh_vin.py, from a zero buffer); one without takes dmsh_pull's layout
// block at buffer 16 with the same vertex buffer as its one stream (RGBA32F / RGBA32UI / RGBA32I, the words Metal's fetch gives).
// Build: clang -fobjc-arc -O1 -framework Metal -framework Foundation proto_vs_pos.m -o <work>/proto_vs_pos
// Usage: proto_vs_pos raster <triangles> <seed> <fragment source>.metal... <label>=<library>:<entry>...
//        proto_vs_pos capture <vertices> <seed> <label>=<library>:<entry>...
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

enum { W = 512, REGION = 1 << 18, MAXFN = 16, MAXPASS = 8, NCOLOR = 8, CAPW = 144 };
typedef struct { uint64_t base; uint32_t stride, divisor; } vstream;
typedef struct { vstream s[16]; uint32_t attr[32]; } vlayout;   // dmsh_pull.metal dm_vlayout
typedef struct { uint64_t dst, pad; uint32_t info[4]; uint32_t stride; int16_t slot[CAPW]; } capture_block;   // nvn_geometry.h nvn_geo_capture
typedef struct { uint64_t ib; float vp[2]; float width; uint32_t prim, count; int32_t base; uint32_t isz, restart, ri, zclip, rect; } wline_block;   // dmsh_pull.py dm_wline (nvn_wline.inc wl_block)
static uint64_t rng, g_seed;
static void reseed(uint64_t salt) { rng = (g_seed * 1000003ull + salt + 1) * 0x9E3779B97F4A7C15ull | 1; }   // each kind of data its own stream, so the uniforms do not change with the number of vertices
static uint32_t rnd(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)(rng >> 16); }
static float rndf(void) { return (float)(rnd() % 2000001) / 1000000.0f - 1.0f; }
// The seed also picks the ranges, so a shader whose path depends on its data (a particle that is drawn only while a uniform time is past an input) finds data it draws with
// (proto_vs_pos.py tries the seeds): uniform floats in [-1, 1], [0, 1] or [-4, 4]; float inputs in [-1, 1], [0, 1], [0, 8] or [-8, 8] (a lifetime is whole frames); integer uniforms 0-3, all 0 or all 1
static int ukind, akind, ikind;
static float rndu(void) { float x = rndf(); return ukind == 1 ? x * 0.5f + 0.5f : ukind == 2 ? x * 4.0f : x; }
static float rnda(void) { float x = rndf(); return akind == 1 ? x * 0.5f + 0.5f : akind == 2 ? x * 4.0f + 4.0f : akind == 3 ? x * 8.0f : x; }
static uint32_t rndi(void) { uint32_t x = rnd() % 4; return ikind == 1 ? 0 : ikind == 2 ? 1 : x; }

static id<MTLDevice> dev;
static id<MTLBuffer> ubuf; static size_t uused; static NSMutableDictionary<NSString *, NSNumber *> *regions;   // each pointer member's region of the uniform data, by its name
static NSMutableDictionary<NSNumber *, id<MTLTexture>> *textures; static id<MTLSamplerState> samp; static NSMutableArray *resources;
static int unsupported;

static int is_int(MTLDataType t) { return (t >= MTLDataTypeInt && t <= MTLDataTypeUInt4) || (t >= MTLDataTypeShort && t <= MTLDataTypeBool4); }
static int comps(MTLDataType t) {
    switch (t) { case MTLDataTypeInt: case MTLDataTypeUInt: case MTLDataTypeBool: return 1; case MTLDataTypeInt2: case MTLDataTypeUInt2: case MTLDataTypeBool2: return 2;
                 case MTLDataTypeInt3: case MTLDataTypeUInt3: case MTLDataTypeBool3: return 3; case MTLDataTypeInt4: case MTLDataTypeUInt4: case MTLDataTypeBool4: return 4; default: return 0; }
}
static void fill_struct(MTLStructType *st, uint8_t *p, uint8_t *end, uint32_t value);
static void fill_type(MTLDataType t, MTLArrayType *at, MTLStructType *st, uint8_t *p, uint8_t *end, uint32_t value) {
    if (t == MTLDataTypeArray) { for (NSUInteger i = 0; i < at.arrayLength && p + i * at.stride + 16 <= end; i++) fill_type(at.elementType, at.elementArrayType, at.elementStructType, p + i * at.stride, end, value); }
    else if (t == MTLDataTypeStruct) fill_struct(st, p, end, value);
    else if (is_int(t)) { for (int c = 0; c < comps(t) && p + 4 * c + 4 <= end; c++) { uint32_t v = value ? value : rndi(); memcpy(p + 4 * c, &v, 4); } }
}
static void fill_struct(MTLStructType *st, uint8_t *p, uint8_t *end, uint32_t value) {
    for (MTLStructMember *m in st.members) fill_type(m.dataType, m.arrayType, m.structType, p + m.offset, end, value);
}
// The region of the uniform data a pointer member reads: random floats, its integer members small integers; a block that is only rows of uint4 holds 0x10000 in every word (the
// sizes the translator's bounded reads compare an index with: 64 KB, a block's largest)
static size_t region_of(MTLStructMember *m) {
    NSNumber *have = regions[m.name]; if (have) return have.unsignedLongValue;
    size_t off = uused; uused += REGION; if (uused > ubuf.length) { fprintf(stderr, "too many buffers\n"); exit(1); }
    uint8_t *p = (uint8_t *)ubuf.contents + off; reseed(100 + regions.count);
    MTLStructType *st = m.pointerType.elementStructType;
    if (st) { int sizes = st.members.count == 1 && st.members[0].dataType == MTLDataTypeArray && st.members[0].arrayType.elementType == MTLDataTypeUInt4 && st.members[0].arrayType.arrayLength <= 64;
              fill_struct(st, p, p + REGION, sizes ? 0x10000 : 0); }
    regions[m.name] = @(off); return off;
}
static id<MTLTexture> texture_of(MTLTextureReferenceType *r) {
    MTLTextureType t = r.textureType; NSNumber *key = @(t * 2 + r.isDepthTexture); id<MTLTexture> x = textures[key]; if (x) return x;
    if (r.isDepthTexture || t == MTLTextureType2DMultisample || t == MTLTextureType1D || t == MTLTextureType1DArray || t == MTLTextureTypeTextureBuffer) { unsupported++; return nil; }
    MTLTextureDescriptor *d = [MTLTextureDescriptor new]; d.textureType = t; d.pixelFormat = MTLPixelFormatRGBA8Unorm; d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModeShared;
    int cube = t == MTLTextureTypeCube || t == MTLTextureTypeCubeArray, arr = t == MTLTextureType2DArray || t == MTLTextureTypeCubeArray, vol = t == MTLTextureType3D;
    d.width = d.height = 32; d.depth = vol ? 8 : 1; d.arrayLength = arr ? 4 : 1;
    x = [dev newTextureWithDescriptor:d];
    uint8_t *px = malloc(32 * 32 * 8 * 4); reseed(1000 + key.unsignedLongValue);
    for (int s = 0; s < (int)d.arrayLength * (cube ? 6 : 1); s++) { for (int i = 0; i < 32 * 32 * 8 * 4; i++) px[i] = (uint8_t)rnd();
        [x replaceRegion:MTLRegionMake3D(0, 0, 0, 32, 32, d.depth) mipmapLevel:0 slice:s withBytes:px bytesPerRow:32 * 4 bytesPerImage:32 * 32 * 4]; }
    free(px); textures[key] = x; [resources addObject:x]; return x;
}
// The argument buffer at buffer 0 of a function, encoded from its reflection: every pointer its region, every texture one of its type, every sampler the one sampler
static id<MTLBuffer> args_of(id<MTLFunction> f, MTLRenderPipelineReflection *refl) {
    MTLArgument *a = nil; for (MTLArgument *x in refl.vertexArguments) if (x.type == MTLArgumentTypeBuffer && x.index == 0) a = x;
    if (!a || a.bufferDataType != MTLDataTypeStruct) return nil;
    id<MTLArgumentEncoder> ae = [f newArgumentEncoderWithBufferIndex:0];
    id<MTLBuffer> b = [dev newBufferWithLength:ae.encodedLength options:MTLResourceStorageModeShared]; [ae setArgumentBuffer:b offset:0];
    for (MTLStructMember *m in a.bufferStructType.members) {
        NSUInteger n = 1; MTLDataType t = m.dataType; MTLTextureReferenceType *tr = m.textureReferenceType;
        if (t == MTLDataTypeArray) { n = m.arrayType.arrayLength; t = m.arrayType.elementType; tr = m.arrayType.elementTextureReferenceType; }
        for (NSUInteger i = 0; i < n; i++) {
            if (t == MTLDataTypePointer) [ae setBuffer:ubuf offset:region_of(m) atIndex:m.argumentIndex + i];
            else if (t == MTLDataTypeTexture) { id<MTLTexture> x = texture_of(tr); if (x) [ae setTexture:x atIndex:m.argumentIndex + i]; }
            else if (t == MTLDataTypeSampler) [ae setSamplerState:samp atIndex:m.argumentIndex + i];
            else unsupported++;
        }
    }
    return b;
}
static id<MTLFunction> function_of(NSString *lib, NSString *entry) {
    NSError *e = nil; id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:lib] error:&e];
    id<MTLFunction> f = l ? [l newFunctionWithName:entry] : nil;
    if (f && f.functionConstantsDictionary.count) {   // the border corrections' constant, off as the host's prebuild sets it; any other keeps its default
        MTLFunctionConstantValues *cv = [MTLFunctionConstantValues new]; BOOL yes = YES;
        if (f.functionConstantsDictionary[@"dm_border_off"]) [cv setConstantValue:&yes type:MTLDataTypeBool withName:@"dm_border_off"];
        f = [l newFunctionWithName:entry constantValues:cv error:&e];
    }
    if (!f) { fprintf(stderr, "%s: no %s (%s)\n", lib.UTF8String, entry.UTF8String, e ? e.localizedDescription.UTF8String : "-"); exit(1); }
    return f;
}

int main(int argc, char **argv) { @autoreleasepool {
    if (argc < 5) { fprintf(stderr, "usage: %s raster|capture <count> <seed> [<fragment>.metal...] <label>=<library>:<entry>...\n", argv[0]); return 2; }
    int capture = !strcmp(argv[1], "capture"), count = atoi(argv[2]), nv = capture ? count : count * 3; uint64_t seed0 = strtoull(argv[3], NULL, 0);
    ukind = (int)(seed0 % 3); akind = (int)(seed0 / 3 % 4); ikind = (int)(seed0 / 12 % 3);
    dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue]; NSError *e = nil;
    regions = [NSMutableDictionary new]; textures = [NSMutableDictionary new]; resources = [NSMutableArray new];
    MTLSamplerDescriptor *sd = [MTLSamplerDescriptor new]; sd.minFilter = sd.magFilter = MTLSamplerMinMagFilterLinear; sd.supportArgumentBuffers = YES; samp = [dev newSamplerStateWithDescriptor:sd];
    NSMutableArray<NSString *> *labels = [NSMutableArray new]; NSMutableArray<id<MTLFunction>> *fns = [NSMutableArray new]; NSMutableArray<id<MTLFunction>> *frags = [NSMutableArray new];
    for (int i = 4; i < argc; i++) {
        NSString *s = @(argv[i]); NSRange eq = [s rangeOfString:@"="];
        if (eq.location == NSNotFound) {   // a fragment pass
            NSString *src = [NSString stringWithContentsOfFile:s encoding:NSUTF8StringEncoding error:&e]; id<MTLLibrary> l = src ? [dev newLibraryWithSource:src options:nil error:&e] : nil;
            if (!l) { fprintf(stderr, "%s: %s\n", argv[i], e.localizedDescription.UTF8String); return 1; }
            [frags addObject:[l newFunctionWithName:@"f0"]]; continue;
        }
        NSString *rest = [s substringFromIndex:eq.location + 1]; NSRange c = [rest rangeOfString:@":" options:NSBackwardsSearch];
        [labels addObject:[s substringToIndex:eq.location]]; [fns addObject:function_of([rest substringToIndex:c.location], [rest substringFromIndex:c.location + 1])];
    }
    int nf = (int)fns.count, np = capture ? 1 : (int)frags.count;
    if (nf < 1 || nf > MAXFN || np < 1 || np > MAXPASS) { fprintf(stderr, "functions or passes out of range\n"); return 2; }
    // the stage inputs, from the first function that has any below attribute 16
    int na = 0, aidx[16], atype[16];
    for (id<MTLFunction> f in fns) { if (na) break;
        for (MTLVertexAttribute *a in f.vertexAttributes) if (a.active && a.attributeIndex < 16 && na < 16) { aidx[na] = (int)a.attributeIndex; atype[na++] = a.attributeType == MTLDataTypeUInt4 ? 1 : a.attributeType == MTLDataTypeInt4 ? 2 : 0; } }
    for (int i = 0; i < na; i++) for (int j = i + 1; j < na; j++) if (aidx[j] < aidx[i]) { int t = aidx[i]; aidx[i] = aidx[j]; aidx[j] = t; t = atype[i]; atype[i] = atype[j]; atype[j] = t; }
    if (!na) { fprintf(stderr, "no function with stage inputs\n"); return 1; }
    g_seed = seed0; reseed(1); int stride = 16 * na;
    id<MTLBuffer> vb = [dev newBufferWithLength:(size_t)nv * stride options:MTLResourceStorageModeShared], zero = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared];
    for (int v = 0; v < nv; v++) for (int a = 0; a < na; a++) for (int c = 0; c < 4; c++) {
        uint8_t *p = (uint8_t *)vb.contents + (size_t)v * stride + a * 16 + c * 4;
        if (atype[a]) { uint32_t x = rnd() % 8; memcpy(p, &x, 4); } else { float x = rnda(); memcpy(p, &x, 4); } }
    ubuf = [dev newBufferWithLength:64 * REGION options:MTLResourceStorageModeShared]; reseed(2);
    for (size_t i = 0; i < ubuf.length / 4; i++) ((float *)ubuf.contents)[i] = rndu();
    vlayout L; memset(&L, 0, sizeof L); L.s[0].base = vb.gpuAddress; L.s[0].stride = stride;
    for (int a = 0; a < na; a++) L.attr[aidx[a]] = (atype[a] == 1 ? 0x2f : atype[a] == 2 ? 0x30 : 0x2e) | (uint32_t)(a * 16) << 8;   // NVN RGBA32UI, RGBA32I, RGBA32F (dm_ftab)
    id<MTLBuffer> lay = [dev newBufferWithBytes:&L length:sizeof L options:MTLResourceStorageModeShared];
    MTLDepthStencilDescriptor *dl = [MTLDepthStencilDescriptor new]; dl.depthCompareFunction = MTLCompareFunctionLess; dl.depthWriteEnabled = YES;
    id<MTLDepthStencilState> less = [dev newDepthStencilStateWithDescriptor:dl];
    id<MTLTexture> color[NCOLOR], depth = nil;
    if (!capture) {
        MTLTextureDescriptor *td = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float width:W height:W mipmapped:NO]; td.usage = MTLTextureUsageRenderTarget; td.storageMode = MTLStorageModeShared;
        for (int i = 0; i < NCOLOR; i++) color[i] = [dev newTextureWithDescriptor:td];
        MTLTextureDescriptor *dd = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float width:W height:W mipmapped:NO]; dd.usage = MTLTextureUsageRenderTarget; dd.storageMode = MTLStorageModePrivate;
        depth = [dev newTextureWithDescriptor:dd];
    }
    size_t dbytes = (size_t)W * W * 4, cbytes = (size_t)W * W * 16, capbytes = (size_t)nv * CAPW * 4;
    id<MTLBuffer> dread = capture ? nil : [dev newBufferWithLength:dbytes options:MTLResourceStorageModeShared], cap = capture ? [dev newBufferWithLength:capbytes options:MTLResourceStorageModeShared] : nil;
    uint8_t *ref_d[MAXPASS] = {0}, *ref_c[MAXPASS] = {0}, *ref_cap = NULL, *cur_c = capture ? NULL : malloc(cbytes * NCOLOR);
    int ref = 0; while (ref < nf && [labels[ref] hasPrefix:@"@"]) ref++;   // the reference: the first function that is drawn (a label with a leading @ only gives the stage inputs)
    for (int fi = ref; fi < nf; fi++) {
        id<MTLFunction> f = fns[fi]; int pulled = 1, shadow = 0; MTLVertexDescriptor *vd = [MTLVertexDescriptor vertexDescriptor];
        for (MTLVertexAttribute *a in f.vertexAttributes) { if (!a.active) continue;
            MTLVertexFormat fmt = a.attributeType == MTLDataTypeUInt4 ? MTLVertexFormatUInt4 : a.attributeType == MTLDataTypeInt4 ? MTLVertexFormatInt4 : MTLVertexFormatFloat4; int at = -1;
            for (int i = 0; i < na; i++) if (aidx[i] == (int)a.attributeIndex) at = i;
            vd.attributes[a.attributeIndex].format = fmt; pulled = 0;
            if (at >= 0) { vd.attributes[a.attributeIndex].offset = at * 16; vd.attributes[a.attributeIndex].bufferIndex = 30; }
            else { vd.attributes[a.attributeIndex].offset = 0; vd.attributes[a.attributeIndex].bufferIndex = 29; shadow = 1; } }   // a shadow input: zeros, as the host's zero buffer
        if (!pulled) vd.layouts[30].stride = stride;
        if (shadow) { vd.layouts[29].stride = 16; vd.layouts[29].stepFunction = MTLVertexStepFunctionConstant; vd.layouts[29].stepRate = 0; }
        for (int p = 0; p < np; p++) {
            MTLRenderPipelineDescriptor *d = [MTLRenderPipelineDescriptor new]; d.vertexFunction = f; if (!pulled) d.vertexDescriptor = vd;
            if (capture) d.rasterizationEnabled = NO;
            else { d.fragmentFunction = frags[p]; for (int i = 0; i < NCOLOR; i++) d.colorAttachments[i].pixelFormat = MTLPixelFormatRGBA32Float; d.depthAttachmentPixelFormat = MTLPixelFormatDepth32Float; }
            MTLRenderPipelineReflection *refl = nil;
            id<MTLRenderPipelineState> ps = [dev newRenderPipelineStateWithDescriptor:d options:MTLPipelineOptionArgumentInfo | MTLPipelineOptionBufferTypeInfo reflection:&refl error:&e];
            if (!ps) { printf("%s\tpass %d\tpipeline failed: %s\n", labels[fi].UTF8String, p, e.localizedDescription.UTF8String); continue; }
            id<MTLBuffer> args = args_of(f, refl); int wide = 0;
            for (MTLArgument *x in refl.vertexArguments) if (x.type == MTLArgumentTypeBuffer && x.index == 15) wide = 1;   // a wide-line entry point (<entry>_wl): the vertices are line list endpoints, drawn 3 pixels wide
            wline_block wl = { 0, { W, W }, 3.0f, 1, (uint32_t)(nv / 2), 0, 0, 0, 0, 1, 0 };
            if (capture) { uint32_t *w = cap.contents; for (size_t i = 0; i < capbytes / 4; i++) w[i] = 0xdeadbeefu; }
            capture_block cb_ = { cap.gpuAddress, 0, { 0, (uint32_t)nv, 0, 1 }, CAPW, { 0 } }; for (int k = 0; k < CAPW; k++) cb_.slot[k] = (int16_t)k;
            id<MTLCommandBuffer> cb = [q commandBuffer];
            MTLRenderPassDescriptor *rp = [MTLRenderPassDescriptor new];
            if (!capture) {
                for (int i = 0; i < NCOLOR; i++) { rp.colorAttachments[i].texture = color[i]; rp.colorAttachments[i].loadAction = MTLLoadActionClear; rp.colorAttachments[i].clearColor = MTLClearColorMake(0, 0, 0, 0); rp.colorAttachments[i].storeAction = MTLStoreActionStore; }
                rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = MTLLoadActionClear; rp.depthAttachment.clearDepth = 1; rp.depthAttachment.storeAction = MTLStoreActionStore;
            } else { rp.renderTargetWidth = rp.renderTargetHeight = 16; rp.defaultRasterSampleCount = 1; }
            id<MTLRenderCommandEncoder> en = [cb renderCommandEncoderWithDescriptor:rp];
            [en setRenderPipelineState:ps]; if (!capture) [en setDepthStencilState:less];
            if (args) [en setVertexBuffer:args offset:0 atIndex:0];
            [en useResource:ubuf usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
            for (id<MTLTexture> x in resources) [en useResource:x usage:MTLResourceUsageRead stages:MTLRenderStageVertex];
            if (pulled) { [en setVertexBuffer:lay offset:0 atIndex:16]; [en useResource:vb usage:MTLResourceUsageRead stages:MTLRenderStageVertex]; }
            else { [en setVertexBuffer:vb offset:0 atIndex:30]; if (shadow) [en setVertexBuffer:zero offset:0 atIndex:29]; }
            if (capture) { [en setVertexBytes:&cb_ length:sizeof cb_ atIndex:13]; [en useResource:cap usage:MTLResourceUsageWrite stages:MTLRenderStageVertex]; }
            if (wide) [en setVertexBytes:&wl length:sizeof wl atIndex:15];
            static int probed; if (!probed) { probed = 1;   // a probe before the first whole draw: 3 vertices in a command buffer of their own. A loop whose count the random uniforms make huge shows here
                [en drawPrimitives:capture ? MTLPrimitiveTypePoint : MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3]; [en endEncoding];
                CFAbsoluteTime t0 = CFAbsoluteTimeGetCurrent(); [cb commit]; [cb waitUntilCompleted]; double dt = CFAbsoluteTimeGetCurrent() - t0;
                if (cb.error || dt > 2.0) { printf("probe\t%s after %.1f s\n", cb.error ? cb.error.localizedDescription.UTF8String : "slow", dt); return 3; }
                p--; continue;   // the same pass again, whole
            }
            [en drawPrimitives:capture ? MTLPrimitiveTypePoint : MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:wide ? nv / 2 * 6 : nv]; [en endEncoding];
            if (!capture) { id<MTLBlitCommandEncoder> bl = [cb blitCommandEncoder];
                [bl copyFromTexture:depth sourceSlice:0 sourceLevel:0 sourceOrigin:MTLOriginMake(0, 0, 0) sourceSize:MTLSizeMake(W, W, 1) toBuffer:dread destinationOffset:0 destinationBytesPerRow:W * 4 destinationBytesPerImage:dbytes];
                [bl endEncoding]; }
            [cb commit]; [cb waitUntilCompleted];
            if (cb.error) { printf("%s\tpass %d\tGPU error: %s\n", labels[fi].UTF8String, p, cb.error.localizedDescription.UTF8String); continue; }
            if (capture) {
                uint32_t *w = cap.contents;
                if (fi == ref) { ref_cap = malloc(capbytes); memcpy(ref_cap, w, capbytes); }
                if (fi == ref && getenv("PV_DUMP")) [[NSData dataWithBytes:w length:capbytes] writeToFile:@(getenv("PV_DUMP")) atomically:NO];   // the reference's records, to look at
                if (!ref_cap) continue;
                uint32_t *r = (uint32_t *)ref_cap; long diff = 0, dpos = 0, dvert = 0, unwritten = 0, nan = 0, far = 0; uint32_t ulps = 0, pulps = 0;
                for (int v = 0; v < nv; v++) { int any = 0;
                    for (int k = 0; k < CAPW; k++) { uint32_t a = w[(size_t)v * CAPW + k], b = r[(size_t)v * CAPW + k]; unwritten += a == 0xdeadbeefu; if (k < 4 && (a & 0x7f800000u) == 0x7f800000u) nan++;
                        if (a != b) { diff++; any = 1; if (k < 4) dpos++;
                            uint32_t u = (a ^ b) >> 31 ? 0xffffffffu : a > b ? a - b : b - a;   // the distance between two floats of one sign, in units in the last place
                            if (u > ulps) ulps = u; if (k < 4 && u > pulps) pulps = u; far += u > 16; } }
                    dvert += any; }
                printf("%s\tcapture\tvertices %d\twords %ld\tdiffering words %ld\tposition words %ld\tvertices differing %ld\tlargest ulps %u\tlargest position ulps %u\twords over 16 ulps %ld\tunwritten %ld\tposition words not finite %ld\tunsupported %d\n",
                       labels[fi].UTF8String, nv, (long)nv * CAPW, diff, dpos, dvert, ulps, pulps, far, unwritten, nan, unsupported);
                continue;
            }
            const float *dz = dread.contents;
            for (int i = 0; i < NCOLOR; i++) [color[i] getBytes:cur_c + cbytes * i bytesPerRow:W * 16 fromRegion:MTLRegionMake2D(0, 0, W, W) mipmapLevel:0];
            if (fi == ref) { ref_d[p] = malloc(dbytes); memcpy(ref_d[p], dz, dbytes); ref_c[p] = malloc(cbytes * NCOLOR); memcpy(ref_c[p], cur_c, cbytes * NCOLOR); }
            if (!ref_d[p]) continue;
            const float *rz = (const float *)ref_d[p]; long covered = 0, ddiff = 0, cdiff = 0; double dmax = 0, cmax = 0; int cmask = 0;
            for (int i = 0; i < W * W; i++) {
                covered += dz[i] != 1.0f;
                if (memcmp(&dz[i], &rz[i], 4)) { ddiff++; double x = fabs((double)dz[i] - rz[i]); if (x > dmax) dmax = x; }
                int any = 0;
                for (int c = 0; c < NCOLOR; c++) { const float *a = (const float *)(cur_c + cbytes * c) + (size_t)i * 4, *b = (const float *)(ref_c[p] + cbytes * c) + (size_t)i * 4;
                    if (memcmp(a, b, 16)) { any = 1; cmask |= 1 << c; for (int k = 0; k < 4; k++) { double x = fabs((double)a[k] - b[k]); if (x > cmax) cmax = x; } } }
                cdiff += any; }
            printf("%s\tpass %d\tcovered %ld\tdepth pixels differing %ld\tlargest depth difference %g\toutput pixels differing %ld\ttargets 0x%x\tlargest output difference %g\tunsupported %d\n",
                   labels[fi].UTF8String, p, covered, ddiff, dmax, cdiff, cmask, cmax, unsupported);
        }
    }
    return 0;
} }
