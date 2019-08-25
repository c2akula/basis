package nd

import "testing"

func TestCopy(t *testing.T) {
	a := Rand(Shape{3, 4, 5})
	b := Rand(Shape{3, 4, 5})
	exp := Zeroslike(a)

	copy(exp.Data(), b.Data()) // exp <- b
	Copy(a, b)                 // a <- b

	if exp.String() != a.String() {
		t.Logf("test 'Copy' failed. exp: %v\n, got: %v\n", exp, a)
		t.Fail()
	}

	av := a.View(Index{1, 0, 1}, Shape{2, 1})
	// fmt.Println(av)
	bv := b.View(Index{1, 1, 1}, Shape{2, 1})
	// fmt.Println(bv)
	exp = Zeroslike(bv)
	for it := exp.Take(); !it.Done(); it.Next() {
		*it.Upk() = bv.Get(it.I())
	}
	// fmt.Println(exp)
	Copy(av, bv) // av <- bv
	if exp.String() != av.String() {
		t.Logf("test 'Copy' failed. exp: %v\n, got: %v\n", exp, av)
		t.Fail()
	}

}

func BenchmarkCopy(b *testing.B) {
	b.ReportAllocs()
	dst := Rand(TestArrayShape)
	src := Rand(TestArrayShape)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Copy(dst, src)
	}
}
