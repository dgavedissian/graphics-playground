#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 colour;
};

vertex VertexOut vertexShader(uint vertexID [[vertex_id]], constant simd::float3* vertexPositions) {
    VertexOut o;
    o.position = float4(vertexPositions[vertexID][0], vertexPositions[vertexID][1], vertexPositions[vertexID][2], 1.0f);
    o.colour = float2(vertexPositions[vertexID][0], vertexPositions[vertexID][1]);
    return o;
}

fragment float4 fragmentShader(VertexOut in [[stage_in]]) {
    return float4(in.colour.x, in.colour.y, 0.0f, 1.0f);
}
