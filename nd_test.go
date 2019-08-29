package go_nd

import "github.com/c2akula/go.nd/nd"

// var TestArrayShape = nd.Shape{100, 100, 100}
// var TestArrayShape = nd.Shape{1, 1, 10}
// var TestArrayShape = nd.Shape{1, 10, 10}
// var TestArrayShape = nd.Shape{60, 10, 10}
// var TestArrayShape = nd.Shape{10, 10, 100}
// var TestArrayShape = nd.Shape{10, 100, 100}
var TestArrayShape = nd.Shape{100, 100, 100}
// var TestArrayShape = nd.Shape{100, 100, 1000}
// var TestArrayShape = nd.Shape{100, 1000, 1000}
// var TestArrayShape = nd.Shape{10, 45, 30}
// var TestArrayShape = nd.Shape{1, 1e8}

// Log-Likelihood Benchmark - Ref: https://arogozhnikov.github.io/2015/09/08/SpeedBenchmarks.html
// Llh - 1e5 - 3.5173ms, Numpy - 6ms - 1.7x faster (go.nd)
// Llh - 1e6 - 35.773ms, Numpy - 62ms - 1.7x faster (go.nd)
// Llh - 1e7 - 519.898ms, Numpy - 857ms - 1.64x faster (go.nd)
// Llh - 1e8 - 7.474s, Numpy - 9.006766s - 1.2x faster (go.nd)
