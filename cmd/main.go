package main

import (
	"fmt"

	"github.com/c2akula/basis/nd"
)

func sub2ind(strides []int, n ...int) (k int) {
	for i, s := range strides {
		k += n[i] * s
	}
	return
}

func main() {
	a := nd.Rand(nd.Shape{4, 35, 15})
	n := Sub2ind(nd.Index{0, 0, 0}, a.Strides())
	fmt.Println("n: ", n)
	// start := time.Now()
	// const N = 1e6

	// for i := 0; i < N; i++ {
	// 	nd.Scale(a.Iter(), 2)
	// }

	// fmt.Println("elapsed: ", time.Since(start)/N)
}
