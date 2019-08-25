package nd

import (
	"fmt"
	"testing"
)

func TestScale(t *testing.T) {
	// a := Reshape(Arange(0, 60), Shape{3,4,5})
	a := Reshape(Arange(0, float64(computeSize(TestArrayShape))), TestArrayShape)
	v := 2.0

	exp := Zeroslike(a)
	Copy(exp, a)
	Apply(exp.Take(), func(f float64) float64 {
		return f * v
	})

	b := Zeroslike(a)
	Copy(b, a)
	Scale(b, v)

	bit, eit := b.Take(), exp.Take()
	for !bit.Done() {
		if *eit.Upk() != *bit.Upk() {
			t.Logf("test failed. exp: %v\n, got: %v\n", exp, b)
			t.Fail()
		}
		bit.Next()
		eit.Next()
	}
}

func TestScaleView(t *testing.T) {
	av := Reshape(Arange(0, 60), Shape{3, 4, 5}).View(Index{1, 0, 1}, Shape{2, 2, 3})
	fmt.Println(av)
	v := 2.0

	exp := Zeroslike(av)
	Copy(exp, av)
	Apply(exp.Take(), func(f float64) float64 {
		return f * v
	})

	b := Zeroslike(av)
	Copy(b, av)
	Scale(b, v)

	bit, eit := b.Take(), exp.Take()
	for !bit.Done() {
		if *eit.Upk() != *bit.Upk() {
			t.Logf("test failed. exp: %v\n, got: %v\n", exp, b)
			t.Fail()
		}
		bit.Next()
		eit.Next()
	}
}

func BenchmarkScale(b *testing.B) {
	x := Rand(TestArrayShape)
	v := 1.1
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Scale(x, v)
	}
}
