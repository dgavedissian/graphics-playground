__kernel void process_array(__global int* data, __global int* out_data) {
    out_data[get_global_id(0)] = data[get_global_id(0)] * 2;
}