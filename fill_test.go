package basis

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/c2akula/basis/nd"
)

func TestFill2(t *testing.T) {
	shp := nd.Shape{3, 2, 3, 2, 5}
	x := nd.Reshape(nd.Arange(0, float64(nd.ComputeSize(shp))), shp)
	fmt.Println(x)
	// xshp := x.Shape()
	// xstr := x.Strides()
	// ndims := x.Ndims()
	// xshpnd := xshp[:ndims-2]
	// xstrnd := xstr[:ndims-2]
	// step := nd.ComputeSize(nd.ComputeStrides(xshpnd))
	// step := xstrnd[ndims-3]
	// fmt.Println("step: ", nd.ComputeStrides(xshpnd))
	// fmt.Println(nd.ComputeSize(xshpnd), xstrnd)
	//
	// for i := 0; i < nd.ComputeSize(xshpnd); i++ {
	// 	fmt.Println("begin: ", i*step)
	// }

	// view - step size equal to stride product of [ndims-2:]
	// # of loops = shape product of [:ndims-2]
	y := x.View(nd.Index{2, 1, 1, 0, 1}, nd.Shape{1, 1, 4})
	// yshp := y.Shape()
	// ystr := y.Strides()
	// ndims = y.Ndims()
	// yshpnd := yshp[:ndims-2]
	// ystrnd := ystr[:ndims-2]
	// step = ystrnd[ndims-3]
	// fmt.Println(nd.ComputeSize(yshpnd), ystrnd)

	// for i := 0; i < nd.ComputeSize(yshpnd); i++ {
	// 	fmt.Println("begin: ", i*step)
	// }

	Fill(2, x)
	Fill(-1, y)
	fmt.Println("y: ", y)

	fmt.Println("x: ", x)
}

func TestFill(t *testing.T) {
	got := nd.Zeros(nd.Shape{3, 4, 5})
	exp := nd.Zeroslike(got)
	ed, ei := exp.Range().Iter()
	for _, k := range ei {
		ed[k] = 1
	}

	gd, gi := got.Range().Iter()
	Fill(1, got)

	for _, k := range gi {
		if ed[k] != gd[k] {
			t.Logf("test 'Fill' failed. exp: %v\n, got: %v\n", exp, got)
			t.Fail()
		}
	}

	gv := got.View(nd.Index{1, 0, 1}, nd.Shape{2, 3})
	gd, gi = gv.Range().Iter()
	ev := exp.View(nd.Index{1, 1, 1}, nd.Shape{2, 3})
	ed, ei = ev.Range().Iter()
	for _, k := range ei {
		ed[k] = 2
	}
	Fill(2, gv)
	for _, k := range gi {
		if ed[k] != gd[k] {
			t.Logf("test 'Fill' failed. exp: %v\n, got: %v\n", ev, gv)
			t.Fail()
		}
	}
}

var A = rand.Float64()

func BenchmarkFill(b *testing.B) {
	b.ReportAllocs()
	// x := nd.Rand(TestArrayShape)
	x := nd.Rand(nd.Shape{1e3, 1e3})
	// xv := x.View(nd.Index{33, 33, 33}, nd.Shape{33, 33, 33})
	a := A
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// Fill(a, xv)
		Fill(a, x)
	}
}

func BenchmarkFill2(b *testing.B) {
	b.ReportAllocs()
	x := nd.Rand(TestArrayShape).Range()
	a := A
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		fill(a, x)
	}
}
