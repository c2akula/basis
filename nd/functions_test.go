package nd

import (
	"math/rand"
	"testing"
)

/*
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
*/

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
