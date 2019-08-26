package go_nd

import (
	"testing"

	"github.com/c2akula/go.nd/nd"
	"github.com/c2akula/go.nd/nd/iter"
)

func TestCopy(t *testing.T) {
	a := nd.Zeros(nd.Shape{3, 4, 5})
	b := nd.Rand(nd.Shape{3, 4, 5})
	exp := nd.Zeroslike(a)

	copy(exp.Data(), b.Data()) // exp <- b
	ait, _, _ := iter.New(a)
	bit, _, _ := iter.New(b)
	Copy(ait, bit) // a <- b

	if exp.String() != a.String() {
		t.Logf("test 'Copy' failed. exp: %v\n, got: %v\n", exp, a)
		t.Fail()
	}

	av := a.View(nd.Index{1, 0, 1}, nd.Shape{2, 1})
	// fmt.Println(av)
	ait, _, _ = iter.New(av)

	bv := b.View(nd.Index{1, 1, 1}, nd.Shape{2, 1})
	// fmt.Println(bv)
	bit, _, _ = iter.New(bv)
	bd := bit.Data()
	bi := bit.Ind()

	exp = nd.Zeroslike(bv)
	eit, ed, ei := iter.New(exp)
	for k := 0; k < eit.Len(); k++ {
		ed[ei[k]] = bd[bi[k]]
	}

	// fmt.Println(exp)
	Copy(ait, bit) // av <- bv
	if exp.String() != av.String() {
		t.Logf("test 'Copy' failed. exp: %v\n, got: %v\n", exp, av)
		t.Fail()
	}
}

func BenchmarkCopy(b *testing.B) {
	b.ReportAllocs()
	dst := nd.Rand(TestArrayShape)
	dit, _, _ := iter.New(dst)
	src := nd.Rand(TestArrayShape)
	sit, _, _ := iter.New(src)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Copy(dit, sit)
	}
}
