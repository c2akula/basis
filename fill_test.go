package go_nd

import (
	"math/rand"
	"testing"

	"github.com/c2akula/go.nd/nd"
	"github.com/c2akula/go.nd/nd/iter"
)

func TestFill(t *testing.T) {
	got := nd.Zeros(nd.Shape{3, 4, 5})
	exp := nd.Zeroslike(got)
	_, ed, ei := iter.New(exp)
	for _, k := range ei {
		ed[k] = 1
	}

	git, gd, gi := iter.New(got)
	Fill(git, 1)

	for _, k := range gi {
		if ed[k] != gd[k] {
			t.Logf("test 'Fill' failed. exp: %v\n, got: %v\n", exp, got)
			t.Fail()
		}
	}

	gv := got.View(nd.Index{1, 0, 1}, nd.Shape{2, 3})
	git, gd, gi = iter.New(gv)
	ev := exp.View(nd.Index{1, 1, 1}, nd.Shape{2, 3})
	eit, _, _ := iter.New(ev)
	ed = eit.Data()
	ei = eit.Ind()
	for _, k := range ei {
		ed[k] = 2
	}
	Fill(git, 2)
	for _, k := range gi {
		if ed[k] != gd[k] {
			t.Logf("test 'Fill' failed. exp: %v\n, got: %v\n", ev, gv)
			t.Fail()
		}
	}
}

func BenchmarkFill(b *testing.B) {
	b.ReportAllocs()
	a := nd.Rand(TestArrayShape)
	it, _, _ := iter.New(a)
	v := rand.Float64()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Fill(it, v)
	}
}
