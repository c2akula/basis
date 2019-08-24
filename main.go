package main

import (
	"fmt"
	"time"

	"local/go.nd/nd"
)

func sub2ind(strides []int, n ...int) (k int) {
	for i, s := range strides {
		k += n[i] * s
	}
	return
}

func main() {
	a := nd.Rand(nd.Shape{4, 35, 15})
	start := time.Now()
	const N = 1e6

	for i := 0; i < N; i++ {
		nd.Scale(a.Take(), 2)
	}

	fmt.Println("elapsed: ", time.Since(start)/N)

}