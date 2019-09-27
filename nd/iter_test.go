package nd

import (
	"fmt"
	"testing"
)

func TestIterator(t *testing.T) {
	shp := Shape{1, 3, 2, 2, 5}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	fmt.Println("x: ", x)
	s := 0.0
	for it := NewIter(x); it.Next(); {
		n, v, _ := it.Get()
		for _, e := range v[:n] {
			s += e * e
		}
	}
	fmt.Println("s: ", s)
}

func TestInd(t *testing.T) {
	shp := Shape{1, 3, 2, 2, 5}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	y := Arange(0, float64(ComputeSize(Shape{1, 2, 2, 1}))).Reshape(Shape{1, 2, 2, 1}) //.Permute([]int{1, 0, 2, 3, 4})
	if err := Broadcast(x, y); err != nil {
		panic(err)
	}
	fmt.Println("x: ", x, "y: ", y)

	it := NewIter(x)
	xit := NewIter(x)
	yit := NewIter(y)
	for k := 0; k < it.size; k++ {
		// ind(it, k)
		xs, ys := ZipInd(xit, yit, k)
		fmt.Println(it.Sub2ind(it.Ind2sub(k)), xs, ys)
	}
}

func BenchmarkInd(b *testing.B) {
	b.Run("new ind", func(b *testing.B) {
		b.ReportAllocs()
		x := Rand(Shape{10, 10, 10, 1e2, 10})
		it := NewIter(x)
		s := 0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for k := 0; k < it.size; k++ {
				s = it.Ind(k)
			}
		}
		_ = s * s
	})

	b.Run("new ind->sub->ind", func(b *testing.B) {
		b.ReportAllocs()
		x := Rand(Shape{10, 10, 10, 1e2, 10})
		it := NewIter(x)
		s := 0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for k := 0; k < it.size; k++ {
				s = it.Sub2ind(it.Ind2sub(k))
			}
		}
		_ = s * s
	})

	b.Run("zipind", func(b *testing.B) {
		b.ReportAllocs()
		x := Rand(Shape{10, 10, 10, 1e2, 10})
		y := Rand(Shape{10, 10, 10, 1e2, 10})
		xit := NewIter(x)
		yit := NewIter(y)
		xs, ys := 0, 0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for k := 0; k < xit.size; k++ {
				xs, ys = ZipInd(xit, yit, k)
			}
		}
		_ = xs * ys
	})

	b.Run("old ind", func(b *testing.B) {
		b.ReportAllocs()
		x := Rand(Shape{10, 10, 10, 1e2, 10})
		it := NewIter(x)
		ind := func(it *Iter, k int) (s int) {
			for i, n := range it.stri {
				j := int(float64(k) * n)
				s += j * it.xstr[i]
				k -= j * it.istr[i]
			}
			return
		}
		s := 0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			for k := 0; k < it.size; k++ {
				s = ind(it, k)
			}
		}
		_ = s * s
	})
}

func BenchmarkIterator(b *testing.B) {
	b.ReportAllocs()
	x := Rand(Shape{10, 10, 10, 1e2, 10})
	it := NewIter(x)
	// a := raFloat64()
	// s := 0.0
	// n := x.Shape()[x.Ndims()-1]
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// s := 0.0
		it.Reset()
		for it.Next() {
			n, v, _ := it.Get()
			for j := range v[:n] {
				v[j] *= v[j]
			}
		}
	}
	// _ = s * s
}

func BenchmarkIteratorFold(b *testing.B) {
	b.Run("internal_fold(*iter)", func(b *testing.B) {
		b.ReportAllocs()
		// x := Rand(Shape{10, 10, 10, 10, 1e2})
		// x := Rand(Shape{1e2, 1e2, 1e2})
		x := Rand(Shape{10, 1e2})
		it := NewIter(x)
		fold := func(it *Iter, memo float64) (s float64) {
			s = memo
			for k := 0; k < it.size; k++ {
				for _, e := range it.x[it.Ind(k):][:it.rn] {
					s += e * e
				}
			}
			return
		}
		s := 0.0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s = fold(it, 0)
		}
		_ = s * s
	})

	b.Run("external_iter_fold", func(b *testing.B) {
		b.ReportAllocs()
		// x := Rand(Shape{10, 10, 10, 10, 1e2})
		// x := Rand(Shape{1e2, 1e2, 1e2})
		x := Rand(Shape{10, 1e2})
		it := NewIter(x)
		s := 0.0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s = 0
			for k := 0; k < it.size; k++ {
				for _, e := range it.x[it.Ind(k):][:it.rn] {
					s += e * e
				}
			}
		}
		_ = s * s
	})

	b.Run("external_iter_fold_next", func(b *testing.B) {
		b.ReportAllocs()
		// x := Rand(Shape{10, 10, 10, 10, 1e2})
		// x := Rand(Shape{1e2, 1e2, 1e2})
		x := Rand(Shape{10, 1e2})
		it := NewIter(x)
		s := 0.0
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			s = 0
			for it.Reset(); it.Next(); {
				n, v, _ := it.Get()
				for _, e := range v[:n] {
					s += e * e
				}
			}
		}
		_ = s * s
	})
}

func TestZip(t *testing.T) {
	shp := Shape{1, 3, 2, 2, 5}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	y := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	fmt.Println("x: ", x, "y: ", y)

	s := 0.0
	for it := Zip(x, y); it.Next(); {
		it.xic = false
		it.yic = false
		xv, yv := it.Get()
		for j, x := range xv.Data[:xv.Size] {
			s += x * yv.Data[j]
		}
	}
	fmt.Println("s: ", s)
}

func BenchmarkZip(b *testing.B) {
	b.Run("zip_at", func(b *testing.B) {

		x := Rand(Shape{10, 10, 10, 10, 1e2})
		y := Rand(Shape{10, 10, 10, 10, 1e2})
		xit := NewIter(x)
		yit := NewIter(y)

		for i := 0; i < b.N; i++ {
			s := 0.0
			for k := 0; k < xit.size; k++ {
				n, xv, _ := xit.At(k)
				_, yv, _ := yit.At(k)

				for j, x := range xv[:n] {
					s += x * yv[j]
				}
			}
		}
	})

	b.Run("zip_ind", func(b *testing.B) {
		x := Rand(Shape{10, 10, 10, 10, 1e2})
		y := Rand(Shape{10, 10, 10, 10, 1e2})
		xit := NewIter(x)
		yit := NewIter(y)
		rn := xit.rn
		for i := 0; i < b.N; i++ {
			s := 0.0
			for k := 0; k < xit.size; k++ {
				xs, ys := ZipInd(xit, yit, k)
				xv := xit.x[xs:]
				yv := yit.x[ys:]
				for j, x := range xv[:rn] {
					s += x * yv[j]
				}
			}
		}
	})

	b.Run("zip", func(b *testing.B) {
		x := Rand(Shape{10, 10, 10, 10, 1e2})
		y := Rand(Shape{10, 10, 10, 10, 1e2})
		it := Zip(x, y)
		for i := 0; i < b.N; i++ {
			s := 0.0
			for it.Reset(); it.Next(); {
				xv, yv := it.Get()
				for j, x := range xv.Data[:xv.Size] {
					s += x * yv.Data[j]
				}
			}
		}
	})
}

func (array *Ndarray) iter() {
	fmt.Println("iter: str = ", array.strides, "shp: ", array.shape)
}

func TestNdarray_Iter(t *testing.T) {
	shp := Shape{3, 4, 5, 6}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp) // [3,4,5,6] => [120,30,6,1]
	fmt.Println("x: ", x)
	xv := x.View(Index{0, 0, 2, 2}, Shape{3, 4, 2, 3})
	// xp := x.Permute([]int{1, 0, 3, 2})
	// xp := x
	xp := xv
	fmt.Println("xp: ", xp)
	xp.iter()
	istr := ComputeStrides(xp.shape)
	fmt.Println("xp.cs: ", istr)

	for it := xp.Iter(); it.Next(); {
		n, v, str := it.Get()
		fmt.Println(it.pind(it.k), n, v, str)
	}

	// the following indexing scheme works quite well for non-View cases
	// find dimension with smallest stride
	// min := func(x []int) (k, v int) {
	// 	v = x[k]
	// 	for i, e := range x[1:] {
	// 		if e < v {
	// 			v = e
	// 			k = i + 1
	// 		}
	// 	}
	// 	return
	// }

	// k, v := min(xp.strides)
	// fmt.Println("k, v: ", k, v)
	// for b := 0; b < xp.size; b += xp.shape[k] {
	// 	fmt.Println(xp.data[b : b+xp.shape[k]])
	// }
}

func Min(x []int) (k, v int) {
	v = x[k]
	for j, e := range x[1:] {
		if e < v {
			v = e
			k = j + 1
		}
	}
	return
}

// fast working version
// func findInd(e, k Index, str Shape) (n int) {
// 	str = str[:len(k)]
// 	e = e[:len(k)]
// 	nd := len(str) - 1

// 	for ; nd >= 0; nd-- {
// 		if k[nd] < e[nd] {
// 			k[nd]++
// 			for i, s := range str {
// 				n += s * k[i]
// 			}
// 			break
// 		}
// 		// reset dimension and do carry
// 		k[nd] = 0
// 	}
// 	return
// }

func findInd(e, k Index, str Shape) (n int) {
	str = str[:len(k)]
	e = e[:len(k)]
	j := len(str) - 1
	for j >= 0 {
		if k[j] < e[j] {
			k[j]++
			for i, s := range str {
				n += s * k[i]
			}
			break
		}
		k[j] = 0
		j--
	}
	return
}

func findIndPlain(size int, i Index, shp, str Shape) (b, n int) {
	k, s := Min(str) // dimension with the smallest stride
	n = shp[k]
	for b := 0; b < size; b += n {
		fmt.Println(b, b+n, s)
	}
	return
}

func TestFindInd(t *testing.T) {
	shp := Shape{3, 4, 5, 6}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp) // [3,4,5,6] => [120,30,6,1]
	fmt.Println("x: ", x)
	// xv := x.View(Index{0, 0, 2, 2}, Shape{3, 4, 2, 3})
	xp := x.Permute([]int{1, 0, 3, 2})
	// xp := x
	// xp := xv
	fmt.Println("xp: ", xp)
	fmt.Println(ComputeStrides(Shape{10, 10, 10, 100, 10}))
	// beg := make(Index, xp.ndims)
	end := make(Index, xp.ndims)
	ComputeEnd(xp.shape, end)
	fmt.Println("end: ", end)
	ind := make(Index, xp.ndims)
	j := 0
	for k := 0; k < ComputeSize(xp.shape[:xp.ndims-1]); k++ {
		// for k := 0; k < ComputeSize(xp.shape[:xp.ndims]); k++ {
		// fmt.Println(ind, j, xp.data[j:])
		// fmt.Println(ind, j, xp.data[j])
		fmt.Println(ind, j)
		// fmt.Println(j)
		// j = findInd(beg[:xp.ndims], end[:xp.ndims], ind[:xp.ndims], xp.strides[:xp.ndims])
		j = findInd(end[:xp.ndims-1], ind[:xp.ndims-1], xp.strides[:xp.ndims-1])
		_ = j
	}
}

func BenchmarkFindInd(b *testing.B) {
	b.ReportAllocs()
	x := Rand(Shape{10, 10, 10, 1e2, 10})
	// beg := make(Index, x.ndims)
	end := make(Index, x.ndims)
	ComputeEnd(x.shape, end)
	ind := make(Index, x.ndims)
	// n := ComputeSize(x.shape[:x.ndims-1])
	b.ResetTimer()
	for i := 0; i < b.N; i++ {

		// for k := 0; k < n; k++ {
		_ = ind
		_ = findInd(end[:x.ndims-1], ind[:x.ndims-1], x.strides[:x.ndims-1])
		// }

	}
}
