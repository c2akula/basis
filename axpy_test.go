package go_nd

import (
	"math/rand"
	"testing"

	"github.com/c2akula/go.nd/nd"
	"github.com/c2akula/go.nd/nd/iter"
)

func TestAxpy(t *testing.T) {
	x := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	y := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	exp := nd.Zeroslike(y)

	xit, xd, xi := iter.New(x)
	yit, yd, yi := iter.New(y)
	_, ed, ei := iter.New(exp)

	for i, k := range ei {
		ed[k] = xd[xi[i]] + yd[yi[i]]
	}

	Axpy(1, xit, yit)

	for i, k := range ei {
		if ed[k] != yd[yi[i]] {
			t.Logf("test 'Axpy' failed. exp: %v\n, got: %v\n", exp, y)
			t.Fail()
		}
	}
}

func TestAxpyView(t *testing.T) {
	xv := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5}).View(nd.Index{1, 0, 1}, nd.Shape{2, 3})
	yv := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5}).View(nd.Index{1, 1, 1}, nd.Shape{2, 3})
	ev := nd.Zeroslike(yv)

	xit, xd, xi := iter.New(xv)
	yit, yd, yi := iter.New(yv)
	_, ed, ei := iter.New(ev)
	for i, k := range ei {
		ed[k] = xd[xi[i]] + yd[yi[i]]
	}

	Axpy(1, xit, yit)

	for i, k := range ei {
		if ed[k] != yd[yi[i]] {
			t.Logf("test 'Axpy' failed. exp: %v\n, got: %v\n", ev, yv)
			t.Fail()
		}
	}
}

func BenchmarkAxpy(bn *testing.B) {
	a := rand.Float64()
	x := nd.Rand(TestArrayShape)
	y := nd.Rand(TestArrayShape)
	xit, _, _ := iter.New(x)
	yit, _, _ := iter.New(y)
	bn.ResetTimer()
	bn.ReportAllocs()
	for i := 0; i < bn.N; i++ {
		Axpy(a, xit, yit)
	}
}
