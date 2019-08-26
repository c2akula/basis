package go_nd

import (
	"strconv"
	"testing"

	"github.com/c2akula/go.nd/nd"
)

func TestDist(t *testing.T) {
	a := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	x := a.View(nd.Index{0, 0, 0}, nd.Shape{4, 5})
	y := a.View(nd.Index{2, 0, 0}, nd.Shape{4, 5})
	exp := "178.8854"
	got := strconv.FormatFloat(Dist(x, y), 'f', 4, 64)
	if exp != got {
		t.Logf("test 'Dist' failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func BenchmarkDist(b *testing.B) {
	b.ReportAllocs()
	x := nd.Rand(TestArrayShape)
	y := nd.Rand(TestArrayShape)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dist(x, y)
	}
}
