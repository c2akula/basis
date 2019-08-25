package nd

import (
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
	// fmt.Println("a: ", a)
	exp := strconv.FormatFloat(-33.2720, 'f', 4, 64)
	got := strconv.FormatFloat(Llh(a.Take(), mu, s), 'f', 4, 64)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}


func TestMax(t *testing.T) {
	a := New(Shape{5, 3}, []float64{
		8, 5, 10,
		2, 4, 13,
		14, 1, 9,
		12, 3, 7,
		6, 11, 0,
	})
	expMax, expK := 14.0, 6
	v, k := Max(a.Take())
	if expMax != v || expK != k {
		t.Logf("test failed. exp=[max: %v, k: %v], got=[max: %v, k: %v]\n", expMax, expK, v, k)
		t.Fail()
	}
}

func TestMin(t *testing.T) {
	a := New(Shape{5, 3}, []float64{
		8, 5, 10,
		2, 4, 13,
		14, 1, 9,
		12, 3, 7,
		6, 11, 0,
	})
	expMin, expK := 0.0, 14
	v, k := Min(a.Take())
	if expMin != v || expK != k {
		t.Logf("test failed. exp=[min: %v, k: %v], got=[min: %v, k: %v]\n", expMin, expK, v, k)
		t.Fail()
	}
}

func TestAllAny(t *testing.T) {
	a := New(Shape{3, 2}, []float64{
		0, 1,
		1, 1,
		0, 0,
	})

	exp := Index{1, 2, 3}
	got := All(a.Take(), func(f float64) bool {
		return f != 0
	})
	for i, v := range exp {
		if got[i] != v {
			t.Logf("test 'where' failed. exp: %v, got: %v\n", exp, got)
			t.Fail()
		}
	}

	a = New(Shape{3, 5}, []float64{
		0.1658, 0.5351, 0.7393, 0.4241, 0.3141,
		0.9508, 0.9818, 0.2158, 0.2524, 0.2426,
		0.9861, 0.4012, 0.6704, 0.1580, 0.4448,
	})
	expAny := 1
	_, gotAny := Any(a.Take(), func(f float64) bool {
		return f > 0.5
	})
	if expAny != gotAny {
		t.Logf("test 'find' failed. exp: %v, got: %v\n", expAny, gotAny)
		t.Fail()
	}
}

func TestTransform(t *testing.T) {
	a := New(Shape{3, 5}, []float64{
		0.1658, 0.5351, 0.7393, 0.4241, 0.3141,
		0.9508, 0.9818, 0.2158, 0.2524, 0.2426,
		0.9861, 0.4012, 0.6704, 0.1580, 0.4448,
	})
	exp := New(Shape{3, 5}, []float64{
		2.1658, 0.5351, 0.7393, 2.4241, 2.3141,
		0.9508, 0.9818, 2.2158, 2.2524, 2.2426,
		0.9861, 2.4012, 0.6704, 2.1580, 2.4448,
	})

	Transform(a.Take(), func(f float64) bool {
		return f < 0.5
	}, func(f float64) float64 {
		return f + 2
	})
	if exp.String() != a.String() {
		t.Logf("test 'Transform' failed. exp: %v\n, got: %v\n", exp, a)
		t.Fail()
	}
}

func TestDistance(t *testing.T) {
	a := New(Shape{3, 5}, []float64{
		5.0000, 11.0000, 14.0000, 1.0000, 3.0000,
		13.0000, 12.0000, 10.0000, 4.0000, 8.0000,
		9.0000, 6.0000, 2.0000, 7.0000, 0.0000,
	})
	b := New(Shape{3, 5}, []float64{
		1.0000, 3.0000, 0.0000, 8.0000, 4.0000,
		7.0000, 11.0000, 13.0000, 12.0000, 6.0000,
		9.0000, 5.0000, 2.0000, 14.0000, 10.0000,
	})

	exp := "24.2899"
	v := Distance(a.Take(), b.Take())
	if exp != strconv.FormatFloat(v, 'f', 4, 64) {
		t.Logf("test 'Distance' failed. exp: %v, got: %v\n", exp, v)
		t.Fail()
	}
}

// Benchmarks

func BenchmarkSub2ind(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	ind := Index{3, 34, 14}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Sub2ind(a, ind)
	}
	_ = ind[0]
}

func BenchmarkSum(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
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
	dst := Rand(TestArrayShape).Take()
	src := Rand(TestArrayShape).Take()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Copy(dst, src)
	}
	_ = dst.Len()
	_ = src.Len()
}

func BenchmarkScale(b *testing.B) {
	array := Rand(TestArrayShape)
	it := array.Take()
	v := rand.Float64()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Scale(it, v)
	}
}

func BenchmarkStd(b *testing.B) {
	array := Rand(TestArrayShape)
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
	a := Rand(TestArrayShape)
	s := rand.Float64()
	fn := func(v float64) float64 {
		return v * s
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Apply(a.Take(), fn)
	}
}

