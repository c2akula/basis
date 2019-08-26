package go_nd

import (
	"testing"

	"github.com/c2akula/go.nd/nd"
)

func TestVar(t *testing.T) {
	a := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	exp := 305.0
	got := Var(a)
	if got != exp {
		t.Logf("test 'Var' failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	av := a.View(nd.Index{2, 1, 1}, nd.Shape{3, 1})
	exp = 25.0
	got = Var(av)
	if got != exp {
		t.Logf("test 'Var' failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func BenchmarkVar(b *testing.B) {
	b.ReportAllocs()
	a := nd.Rand(TestArrayShape)
	v := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		v = Var(a)
	}
	_ = v * v
}
