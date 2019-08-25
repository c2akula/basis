package nd

import (
	"strconv"
	"testing"
)

func TestDist(t *testing.T) {
	a := Reshape(Arange(0, 60), Shape{3, 4, 5})
	x := a.View(Index{0, 0, 0}, Shape{4, 5})
	y := a.View(Index{2, 0, 0}, Shape{4, 5})
	exp := "178.8854"
	got := strconv.FormatFloat(Dist(x, y), 'f', 4, 64)
	if exp != got {
		t.Logf("test 'Dist' failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func BenchmarkDist(b *testing.B) {
	b.ReportAllocs()
	x := Rand(TestArrayShape)
	y := Rand(TestArrayShape)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dist(x, y)
	}
}
