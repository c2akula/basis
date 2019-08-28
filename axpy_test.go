package go_nd

import (
	"math/rand"
	"testing"

	"github.com/c2akula/go.nd/nd"
)

func TestAxpy(t *testing.T) {
	x := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	y := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	exp := nd.Zeroslike(y)

	xit := x.Iter()
	xd, xi := x.Data(), xit.Ind()

	yit := y.Iter()
	yd, yi := y.Data(), yit.Ind()
	ed, ei := exp.Data(), exp.Iter().Ind()

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

	xit := xv.Iter()
	xd, xi := xv.Data(), xit.Ind()
	yit := yv.Iter()
	yd, yi := yv.Data(), yit.Ind()
	ed, ei := ev.Data(), ev.Iter().Ind()
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
	x := nd.Rand(TestArrayShape).Range()
	y := nd.Rand(TestArrayShape).Range()
	bn.ResetTimer()
	bn.ReportAllocs()
	for i := 0; i < bn.N; i++ {
		Axpy(a, x, y)
	}
}
