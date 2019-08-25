package nd

import "testing"

func TestDot(t *testing.T) {
	x := Reshape(Arange(0, 60), Shape{3, 4, 5})
	exp := 70210.0
	got := Dot(x, x)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y := x.View(Index{0, 0, 0}, Shape{1, 4, 1})
	exp = 350
	got = Dot(y, y)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y = x.View(Index{0, 0, 0}, Shape{2, 4, 1})
	exp = 3500
	got = Dot(y, y)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y = x.View(Index{0, 1, 1}, Shape{3, 3, 4})
	exp = 48270.0
	got = Dot(y, y)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

	y = x.View(Index{1, 1, 1}, Shape{2, 3, 4})
	exp = 46180.0
	got = Dot(y, y)
	if exp != got {
		t.Logf("test failed. exp: %v, got: %v\n", exp, got)
		t.Fail()
	}

}

func BenchmarkDot(b *testing.B) {
	b.ReportAllocs()
	// x := Rand(Shape{100, 50})
	// y := Rand(Shape{100, 50})
	x := Rand(TestArrayShape)
	y := Rand(TestArrayShape)
	f := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		f = Dot(x, y)
	}
	_ = f * f
}
