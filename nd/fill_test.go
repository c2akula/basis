package nd

import (
	"math/rand"
	"testing"
)

func TestFill(t *testing.T) {
	got := Zeros(Shape{3, 4, 5})
	exp := Zeroslike(got)
	for it := exp.Take(); !it.Done(); it.Next() {
		*it.Upk() = 1
	}
	Fill(got, 1)

	for it := exp.Take(); !it.Done(); it.Next() {
		if *it.Upk() != got.Get(it.I()) {
			t.Logf("test 'Fill' failed. exp: %v\n, got: %v\n", exp, got)
			t.Fail()
		}
	}

	gv := got.View(Index{1, 0, 1}, Shape{2, 3})
	ev := exp.View(Index{1, 1, 1}, Shape{2, 3})
	it := Iter(ev)
	for ; !it.Done(); it.Next() {
		*it.Upk() = 2
	}
	it.Reset()
	Fill(gv, 2)
	for ; !it.Done(); it.Next() {
		if *it.Upk() != gv.Get(it.I()) {
			t.Logf("test 'Fill' failed. exp: %v\n, got: %v\n", ev, gv)
			t.Fail()
		}
	}
}

func BenchmarkFill(b *testing.B) {
	b.ReportAllocs()
	a := Rand(TestArrayShape)
	v := rand.Float64()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Fill(a, v)
	}
}
