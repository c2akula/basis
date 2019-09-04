package basis

import (
	"fmt"
	"testing"

	"github.com/c2akula/basis/nd"
)

func TestDot2(t *testing.T) {
	a := nd.Reshape(nd.Arange(0, 60*2), nd.Shape{2, 3, 4, 5})
	b := nd.Reshape(nd.Arange(0, 60*2), nd.Shape{2, 3, 4, 5})
	_ = Dot(a, b)
}

func TestDot(t *testing.T) {
	a := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	b := nd.Reshape(nd.Arange(0, 60), nd.Shape{3, 4, 5})
	exp := 70210.0
	got := Dot(a, b)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y1 := a.View(nd.Index{0, 0, 0}, nd.Shape{1, 4, 1}) // a(:,:,0)
	y2 := a.View(nd.Index{1, 0, 0}, nd.Shape{1, 4, 1}) // a(:,:,1)
	exp = 950.0
	got = Dot(y1, y2)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y3 := a.View(nd.Index{0, 0, 0}, nd.Shape{2, 4, 1})
	fmt.Println("y3:\n", y3)
	exp = 3500
	got = Dot(y3, y3)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y4 := a.View(nd.Index{0, 1, 1}, nd.Shape{3, 3, 4})
	exp = 48270.0
	got = Dot(y4, y4)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y5 := a.View(nd.Index{1, 1, 1}, nd.Shape{2, 3, 4})
	exp = 46180.0
	got = Dot(y5, y5)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}
}

func BenchmarkDot(b *testing.B) {
	b.ReportAllocs()
	x := nd.Rand(TestArrayShape)
	y := nd.Rand(TestArrayShape)
	f := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f = Dot(x, y)
	}
	_ = f * f
}

func BenchmarkDot2(b *testing.B) {
	b.ReportAllocs()
	x := nd.Rand(TestArrayShape).Range()
	y := nd.Rand(TestArrayShape).Range()
	f := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f = dot(x, y)
	}
	_ = f * f
}
