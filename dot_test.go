package go_nd

import (
	"testing"

	"github.com/c2akula/go.nd/nd"
	"github.com/c2akula/go.nd/nd/iter"
)

func TestDot(t *testing.T) {
	a := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	x, _, _ := iter.New(a)
	exp := 70210.0
	got := Dot(x, x)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y, _, _ := iter.New(a.View(nd.Index{0, 0, 0}, nd.Shape{1, 4, 1}))
	exp = 350
	got = Dot(y, y)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y, _, _ = iter.New(a.View(nd.Index{0, 0, 0}, nd.Shape{2, 4, 1}))
	exp = 3500
	got = Dot(y, y)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y, _, _ = iter.New(a.View(nd.Index{0, 1, 1}, nd.Shape{3, 3, 4}))
	exp = 48270.0
	got = Dot(y, y)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y, _, _ = iter.New(a.View(nd.Index{1, 1, 1}, nd.Shape{2, 3, 4}))
	exp = 46180.0
	got = Dot(y, y)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func BenchmarkDot(b *testing.B) {
	b.ReportAllocs()
	x, _, _ := iter.New(nd.Rand(TestArrayShape))
	y, _, _ := iter.New(nd.Rand(TestArrayShape))
	f := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f = Dot(x, y)
	}
	_ = f * f
}
