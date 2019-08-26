package go_nd

import (
	"math/rand"
	"testing"

	"github.com/c2akula/go.nd/nd"
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

func TestAllAny(t *testing.T) {
	a := nd.New(nd.Shape{3, 2}, []float64{
		0, 1,
		1, 1,
		0, 0,
	})

	exp := nd.Index{1, 2, 3}
	got := All(a.Take(), func(f float64) bool {
		return f != 0
	})
	for i, v := range exp {
		if got[i] != v {
			t.Logf("test 'where' failed. exp: %v, got: %v\n", exp, got)
			t.Fail()
		}
	}

	a = nd.New(nd.Shape{3, 5}, []float64{
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
	a := nd.New(nd.Shape{3, 5}, []float64{
		0.1658, 0.5351, 0.7393, 0.4241, 0.3141,
		0.9508, 0.9818, 0.2158, 0.2524, 0.2426,
		0.9861, 0.4012, 0.6704, 0.1580, 0.4448,
	})
	exp := nd.New(nd.Shape{3, 5}, []float64{
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
	a := nd.Rand(TestArrayShape)
	strides := a.Strides()
	ind := nd.Index{3, 34, 14}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		nd.Sub2ind(strides, ind)
	}
	_ = ind[0]
}

func BenchmarkApply(b *testing.B) {
	b.ReportAllocs()
	a := nd.Rand(TestArrayShape)
	s := rand.Float64()
	fn := func(v float64) float64 {
		return v * s
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Apply(a.Take(), fn)
	}
}
