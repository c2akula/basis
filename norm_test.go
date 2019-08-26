package go_nd

import (
	"strconv"
	"testing"

	"github.com/c2akula/go.nd/nd"
	"github.com/c2akula/go.nd/nd/iter"
)

func TestNorm(t *testing.T) {
	a := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	it, _, _ := iter.New(a)
	exp := "264.9717"
	got := strconv.FormatFloat(Norm(it), 'f', 4, 64)
	if got != exp {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func BenchmarkNorm(b *testing.B) {
	a := nd.Rand(TestArrayShape)
	it, _, _ := iter.New(a)
	b.ResetTimer()
	b.ReportAllocs()
	n := 0.0
	for i := 0; i < b.N; i++ {
		n = Norm(it)
	}
	_ = n * n
}
