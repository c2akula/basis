package go_nd

import (
	"testing"

	"github.com/c2akula/go.nd/nd"
)

func TestCopy(t *testing.T) {
	a := nd.Zeros(nd.Shape{3, 4, 5})
	b := nd.Rand(nd.Shape{3, 4, 5})
	exp := nd.Zeroslike(a)

	copy(exp.Data(), b.Data()) // exp <- b
	ait := a.Range()
	bit := b.Range()
	Copy(ait, bit) // a <- b

	if exp.String() != a.String() {
		t.Logf("test 'Copy' failed. exp: %v\n, got: %v\n", exp, a)
		t.Fail()
	}

	av := a.View(nd.Index{1, 0, 1}, nd.Shape{2, 1})
	// fmt.Println(av)
	ait = av.Range()

	bv := b.View(nd.Index{1, 1, 1}, nd.Shape{2, 1})
	// fmt.Println(bv)
	bit = bv.Range()
	bd := bit.Data()
	bi := bit.Ind()

	exp = nd.Zeroslike(bv)
	eit := exp.Iter()
	ed := exp.Data()
	ei := eit.Ind()
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
	dit := dst.Range()
	src := nd.Rand(TestArrayShape)
	sit := src.Range()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Copy(dit, sit)
	}
}
