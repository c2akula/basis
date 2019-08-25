package nd

import (
	"strconv"
	"testing"
)

func TestNorm(t *testing.T) {
	a := Reshape(Arange(0, 60), Shape{3, 4, 5})
	exp := "264.9717"
	got := strconv.FormatFloat(Norm(a), 'f', 4, 64)
	if got != exp {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func BenchmarkNorm(b *testing.B) {
	a := Rand(TestArrayShape)
	b.ResetTimer()
	b.ReportAllocs()
	n := 0.0
	for i := 0; i < b.N; i++ {
		n = Norm(a)
	}
	_ = n * n
}
