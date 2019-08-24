package nd

import (
	"fmt"
	"math/rand"
	"strconv"
	"testing"
)

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
	got := Sum(a.Take())
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
	}
	b := a.View(
		Index{1, 0, 1, 0},
		Shape{1, 2, 1, 3},
	)
	exp = 9.0
	it := Iter(b)
	got = Sum(it)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func TestCopy(t *testing.T) {
	a := Rand(Shape{4, 35, 15})
	b := Rand(Shape{35, 60})
	exp := Zeroslike(a)
	copy(exp.Data(), b.Data()) // exp <- b
	Copy(a.Take(), b.Take())   // a <- b

	if exp.String() != a.String() {
		t.Logf("test failed. exp: %v\n, got: %v\n", exp, a)
		t.Fail()
	}
}

func TestScale(t *testing.T) {
	a := Reshape(Arange(1, 15), Shape{2, 7})
	v := 2.0

	// scale := func(it Iterator, v float64) {
	// 	for !it.Done() {
	// 		it.Set(it.Get() * v)
	// 		it.Next()
	// 	}
	// 	it.Reset()
	// }
	exp := Zeroslike(a)
	Copy(exp.Take(), a.Take())
	Apply(exp.Take(), func(f float64) float64 {
		return f * v
	})
	// scale(exp.Take(), v)

	b := Zeroslike(a)
	Copy(b.Take(), a.Take())
	Scale(b.Take(), v)

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

func TestMean(t *testing.T) {
	a := Reshape(Arange(1, 15), Shape{2, 7})
	exp := strconv.FormatFloat(7.5, 'f', 4, 64)
	got := strconv.FormatFloat(Mean(a.Take()), 'f', 4, 64)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func TestVar(t *testing.T) {
	a := Reshape(Arange(1, 15), Shape{2, 7})
	exp := strconv.FormatFloat(17.5, 'f', 4, 64)
	got := strconv.FormatFloat(Var(a.Take()), 'f', 4, 64)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func TestStd(t *testing.T) {
	a := Reshape(Arange(1, 15), Shape{2, 7})
	fmt.Println("a: ", a)
	exp := strconv.FormatFloat(4.18330013267038, 'f', 4, 64)
	got := strconv.FormatFloat(Std(a.Take()), 'f', 4, 64)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func TestLlh(t *testing.T) {
	a := Reshape(Arange(1, 15), Shape{2, 7})
	mu := Mean(a.Take())
	s := Var(a.Take())
	fmt.Println(Llh(a.Take(), mu, s))
	// fmt.Println("a: ", a)
	exp := strconv.FormatFloat(-33.2720, 'f', 4, 64)
	got := strconv.FormatFloat(Llh(a.Take(), mu, s), 'f', 4, 64)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

// Benchmarks

func BenchmarkSub2ind(b *testing.B) {
	b.ReportAllocs()
	a := Rand(Shape{4, 35, 15})
	ind := Index{3, 34, 14}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sub2ind(a, ind)
	}
	_ = ind[0]
}

func BenchmarkSum(b *testing.B) {
	b.ReportAllocs()
	a := Rand(Shape{4, 35, 15})
	it := a.Take()
	b.ResetTimer()
	var sum float64
	for i := 0; i < b.N; i++ {
		sum = Sum(it)
	}
	_ = sum * float64(it.Len())
}

func BenchmarkCopy(b *testing.B) {
	b.ReportAllocs()
	dst := Rand(Shape{4, 35, 15}).Take()
	src := Rand(Shape{4, 35, 15}).Take()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Copy(dst, src)
	}
	_ = dst.Len()
	_ = src.Len()
}

func BenchmarkScale(b *testing.B) {
	array := Rand(Shape{4, 35, 15})
	it := array.Take()
	v := rand.Float64()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Scale(it, v)
	}
}

func BenchmarkStd(b *testing.B) {
	array := Rand(Shape{4, 35, 15})
	it := array.Take()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Std(it)
	}
}

func BenchmarkLlh(b *testing.B) {
	array := Rand(Shape{4, 35, 15})
	mean := rand.Float64()
	sigma := rand.Float64()
	l := 0.0
	it := array.Take()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l = Llh(it, mean, sigma)
	}
	_ = l * l
}

func BenchmarkApply(b *testing.B) {
	b.ReportAllocs()
	a := Rand(Shape{4, 35, 15})
	s := rand.Float64()
	fn := func(v float64) float64 {
		return v * s
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Apply(a.Take(), fn)
	}
}

func TestDot(t *testing.T) {
	x := Arange(0, 15)
	y := Arange(0, 15)
	exp := 1015.0
	got := Dot(x.Take(), y.Take())
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func BenchmarkDot(b *testing.B) {
	b.ReportAllocs()
	x := Rand(Shape{4, 35, 15})
	y := Rand(Shape{4, 35, 15})
	f := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f = Dot(x.Take(), y.Take())
	}
	_ = f * f
}
