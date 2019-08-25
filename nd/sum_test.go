package nd

import "testing"

func TestSum(t *testing.T) {
	a := New(Shape{2, 2, 2, 3}, []float64{
		// t = 0, p = 0
		1, 2, 3, // r = 0, c = 0:2
		4, 5, 6, // r = 1, c = 0:2
		// t = 0, p = 1
		7, 8, 9, // r = 0, c = 0:2
		2, 0, 1, // r = 1, c = 0:2

		// t = 1, p = 0
		6, 4, 5,
		3, 1, 2,
		// t = 1, p = 1
		9, 7, 8,
		1, 0, 2,
	})
	exp := 96.0
	got := Sum(a)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
	}
	b := a.View(
		Index{1, 0, 1, 0},
		Shape{1, 2, 1, 3},
	)
	exp = 9.0
	got = Sum(b)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func BenchmarkSum(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	b.ResetTimer()
	var sum float64
	for i := 0; i < b.N; i++ {
		sum = Sum(a)
	}
	_ = sum * sum
}
