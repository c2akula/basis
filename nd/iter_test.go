package nd

import (
	"fmt"
	"math/rand"
	"testing"
)

var RandArray = Rand(TestArrayShape)

func BenchmarkIterConcreteRange(b *testing.B) {
	b.ReportAllocs()
	a := RandArray
	ad, ai := a.Range().Iter()
	s := 0.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, k := range ai {
			ad[k] *= 5.5
		}
	}
	_ = s * s
}

func TestIter_Iter(t *testing.T) {
	a := Rand(Shape{3, 4, 5})
	ait := a.Iter()
	for i, k := range ait.Ind() {
		if i != k {
			t.Logf("test 'Iter_Iter' failed. exp: %d, got: %d\n", i, k)
			t.Fail()
		}
	}

	av := a.View(Index{1, 0, 1}, Shape{1, 2, 3})
	ait = av.Iter()
	avstr := ComputeStrides(av.Shape())
	exp := make(Index, av.Size())
	ind := make(Index, av.Ndims())
	for i := range exp {
		exp[i] = Sub2ind(av.Strides(), Ind2sub(avstr, i, ind))
	}
	for i, k := range ait.Ind() {
		if exp[i] != k {
			t.Logf("test 'Iter_Iter' failed. exp: %d, got: %d\n", exp, ait.Ind())
			t.Fail()
		}
	}
}

type iterator struct {
	*ndarray
	beg, len int
	ind      int
}

func TestIter(t *testing.T) {
	x := Reshape(Arange(0, 60), Shape{3, 4, 5})
	y := x.View(Index{1, 0, 1}, Shape{2, 2})
	it := &iterator{y.(*ndarray), 0, 0, 0}
	fmt.Println(it.data)

	next := func(it *iterator) (float64, bool) {
		v := it.data[0]
		it.ind++
		return v, it.ind > it.len
	}

	next(it)
}

type counter struct {
	ndims int
	// b, i, e Index
	b    Index
	i, e int
	n    Shape
	k    int
}

func newcounter(b Index, shape Shape) *counter {
	c := &counter{n: ComputeStrides(shape)}
	c.i = Sub2ind(c.n, b)
	c.k = c.i
	c.b = b
	e := make(Index, len(b))
	for k, n := range shape {
		e[k] = n + b[k] - 1
	}
	c.e = Sub2ind(c.n, e)
	fmt.Println("c.n: ", c.n)
	// c := &counter{
	// 	ndims: len(b),
	// 	b:     b,
	// 	e:     e,
	// 	i:     make(Index, len(b)),
	// 	n:     ComputeStrides(shape),
	// }
	// copy(c.i, c.b)
	return c
}

func (c *counter) inc() {
	c.k++
	/*
		for k := c.ndims - 1; k >= 0; k-- {
			if c.i[k]++; c.i[k] > c.e[k] {
				if k > 0 {
					// fmt.Printf("bef: c.i[k=%d]: %v\n", k, c.i)
					c.i[k] = c.b[k]
					// fmt.Printf("aft: c.i[k=%d]: %v\n", k, c.i)
				}
			} else {
				return
			}
		}
	*/
}

func (c *counter) done() bool {
	// return c.i[0] > c.e[0]
	return c.k > c.e
}

// func (c *counter) at() Index { return c.i }
func (c *counter) reset() {
	// copy(c.i, c.b)
	c.k = c.i
}
func TestCounter(t *testing.T) {
	x := Reshape(Arange(0, 60), Shape{3, 4, 5})
	fmt.Println("x: ", x)
	c := newcounter(Index{1, 0, 1}, Shape{2, 2, 2})
	// c.ind2sub(2)
	for ; !c.done(); c.inc() {
		fmt.Println(c.k)
	}
	// c.inc(2)
	// fmt.Println(c.i)
}

func permute(array *ndarray, order []int) *ndarray {
	res := array.View(make(Index, array.ndims), array.shape).(*ndarray)

	if order == nil {
		order = make([]int, array.ndims)
		for k, j := array.ndims-1, 0; k >= 0; k, j = k-1, j+1 {
			order[j] = k
		}
	} else if len(order) != array.ndims {
		panic("no. of dimensions in the permutation order list != array.ndims")
	}
	/*
		https://stackoverflow.com/a/7366196
		for i in xrange(len(a)):
			x = a[i]
			j = i
			while True:
				k = indices[j]
				indices[j] = j
				if k == i:
					break
				a[j] = a[k]
				j = k
			a[j] = x
	*/

	shp := res.shape
	str := res.strides
	for k := 0; k < res.ndims; k++ {
		xshp := shp[k]
		xstr := str[k]
		j := k
		for {
			i := order[j]
			order[j] = j
			if i == k {
				break
			}
			shp[j] = shp[i]
			str[j] = str[i]
			j = i
		}
		shp[j] = xshp
		str[j] = xstr
	}
	// fmt.Printf("res.shp: %v, res.str: %v\n", res.shape, res.strides)
	return res
}

func transpose(array *ndarray) *ndarray {
	if array.ndims > 2 {
		panic("transpose only defined on 2D Arrays")
	}
	return permute(array, nil)
}

func TestTranspose(t *testing.T) {
	shp := Shape{3, 4, 5}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	// fmt.Println("x: ", x)
	// xt := permute(x, nil)
	// fmt.Println("xt: ", xt)

	xt := permute(x, []int{2, 1, 0})
	fmt.Println("xt: ", xt)

	xv := x.View(Index{1, 0, 1}, Shape{2, 2, 3}).(*ndarray)
	fmt.Println("xv: ", xv)
	xvt := permute(xv, []int{0, 2, 1})
	y := xvt.String()
	fmt.Println("y: ", y)
}

type iter2d struct {
	k           int
	inc, len, n int
	shp, str    [2]int
	arr         []float64
}

func newiter2d(array *ndarray) *iter2d {
	it := &iter2d{
		shp: [2]int{array.shape[0], array.shape[1]},
		str: [2]int{array.strides[0], array.strides[1]},
		arr: array.data,
	}

	if it.str[1] > 1 {
		it.inc = it.str[1]
		it.len = it.shp[1]
		it.n = it.shp[0]
	} else {
		it.inc = it.str[0]
		it.len = it.shp[0]
		it.n = it.shp[1]
	}

	return it
}

func (it *iter2d) done() bool { return it.k == it.len }

func (it *iter2d) next() []float64 {
	b := it.k * it.inc
	it.k++
	return it.arr[b : b+it.n]
}

func (it *iter2d) reset() { it.k = 0 }

func TestIter2d(t *testing.T) {
	shp := Shape{3, 4}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	fmt.Println("x: ", x)
	exp := make([]float64, x.size)
	for i := range exp {
		exp[i] = float64(i)
	}

	for it, i := newiter2d(x), 0; !it.done(); {
		for _, v := range it.next() {
			if v != exp[i] {
				t.Logf("test failed. got: %v\nexp: %v\n", it.arr, exp)
				t.Fail()
			}
			i++
		}
	}

	xv := x.View(Index{1, 1}, Shape{2, 3}).(*ndarray)
	fmt.Println("xv: ", xv)
	exp = []float64{5, 6, 7, 9, 10, 11}
	for it, i := newiter2d(xv), 0; !it.done(); {
		for _, v := range it.next() {
			if v != exp[i] {
				t.Logf("test failed. got: %v\nexp: %v\n", it.arr, exp)
				t.Fail()
			}
			i++
		}
	}

	xvt := transpose(xv)
	fmt.Println("xvt: ", xvt)
	exp = []float64{5, 6, 7, 9, 10, 11}
	for it, i := newiter2d(xvt), 0; !it.done(); {
		for _, v := range it.next() {
			if v != exp[i] {
				t.Logf("test failed. got: %v\nexp: %v\n", it.arr, exp)
				t.Fail()
			}
			i++
		}
	}

	dot := func(x, y *iter2d) (s float64) {
		if ComputeSize(x.shp[:]) != ComputeSize(y.shp[:]) {
			panic("dot: dimensions must be same")
		}
		// if (x.shp[0] != y.shp[0]) || (x.shp[1] != y.shp[1]) {
		// }

		if x == y {
			for !x.done() {
				for _, v := range x.next() {
					s += v * v
				}
			}
			x.reset()
			return
		}

		for {
			xv := x.next()
			yv := y.next()
			for k, e := range xv {
				s += e * yv[k]
			}

			if x.done() && y.done() {
				break
			}
		}
		x.reset()
		y.reset()
		return
	}

	it := newiter2d(xv)
	s := dot(it, newiter2d(xvt))
	fmt.Println("s: ", s)
}

func BenchmarkIter2d(b *testing.B) {
	b.ReportAllocs()
	shp := Shape{1e3, 1e3}
	x := Rand(shp).(*ndarray)
	y := Rand(shp).(*ndarray)
	xit := newiter2d(x)
	yit := newiter2d(y)
	a := rand.Float64()
	// fill := func(it *iter2d, a float64) *iter2d {
	// 	for !it.done() {
	// 		v := it.next()
	// 		for i := range v {
	// 			v[i] = a
	// 		}
	// 	}
	// 	it.reset()
	// 	return it
	// }

	dot := func(x, y *iter2d) (s float64) {
		if ComputeSize(x.shp[:]) != ComputeSize(y.shp[:]) {
			panic("dot: dimensions must be same")
		}

		if x == y {
			for !x.done() {
				for _, v := range x.next() {
					s += v * v
				}
			}
			x.reset()
			return
		}

		for {
			xv := x.next()
			yv := y.next()
			for k, e := range xv {
				s += e * yv[k]
			}

			if x.done() && y.done() {
				break
			}
		}
		x.reset()
		y.reset()
		return
	}
	// _ = dot(xit, yit)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// xit = fill(xit, a)
		a = dot(xit, yit)
	}
	// _ = xit.arr[0] * xit.arr[0]
	_ = a * a
}

type nditer struct {
	shp, str Shape
	istr     []float64 // store the inverse of strides
	sub      Index
	k, b     int // indices
	len      int
	arr      *ndarray
	shp2d    [2]int
	str2d    [2]int
	isplain  bool // 2d array's str[1] == 1
}

func newnditer(array *ndarray) *nditer {
	nd := array.ndims
	shp := array.shape
	shpnd := array.shape[:nd-1]
	shp2d := array.shape[nd-2:]
	str2d := array.strides[nd-2:]

	it := &nditer{
		arr:     array,
		shp:     make(Shape, nd),     // shp
		istr:    make([]float64, nd), // istr
		sub:     make(Index, nd),     // sub
		len:     ComputeSize(shpnd),
		isplain: !(str2d[1] > 1),
	}

	copy(it.shp, shp)
	for i := nd - 1; i < nd; i++ {
		it.shp[i] = 1
	}
	copy(it.shp2d[:], shp2d)

	// store the inverse of the strides for faster ind2sub calculation
	it.str = ComputeStrides(it.shp)
	for i, s := range it.str {
		it.istr[i] = 1 / float64(s)
	}
	copy(it.str2d[:], str2d)
	return it
}

func (it *nditer) ind(k int) (s int) {
	for i, n := range it.istr {
		j := int(float64(k) * n)
		s += j * it.arr.strides[i]
		k -= j * it.str[i]
		it.sub[i] = j
	}
	return
}

func (it *nditer) Done() bool {
	return it.k == it.len
}

func (it *nditer) Cursor() int { return it.b }

func (it *nditer) Reset() { it.k = 0 }

func (it *nditer) Inc() int { return it.str2d[1] }

func (it *nditer) Next() ([]float64, int) {
	it.b = it.ind(it.k)
	x := it.arr.data[it.b:]
	it.k++
	return x, it.shp2d[1]
}

func (it *nditer) ZipNext(y *nditer) ([]float64, []float64, int) {
	it.b = it.ind(it.k)
	xv := it.arr.data[it.b:]
	yv := y.arr.data[it.b:]
	it.k++
	return xv, yv, it.shp2d[1]
}

func (it *nditer) Map(fn func(float64) float64) *nditer {
	if !it.isplain {
		for inc := it.Inc(); !it.Done(); {
			x, n := it.Next()
			for i := range x[:n] {
				i *= inc
				x[i] = fn(x[i])
			}
		}
	} else {
		for !it.Done() {
			x, n := it.Next()
			for i, v := range x[:n] {
				x[i] = fn(v)
			}
		}
	}
	it.Reset()
	return it
}

func (it *nditer) Fold(init float64, fn func(float64) float64) (s float64) {
	s = init
	if !it.isplain {
		for inc := it.Inc(); !it.Done(); {
			x, n := it.Next()
			for i := range x[:n] {
				i *= inc
				s += fn(x[i])
			}
		}
		it.Reset()
		return
	}

	for !it.Done() {
		x, n := it.Next()
		for _, v := range x[:n] {
			s += fn(v)
		}
	}
	it.Reset()
	return
}

func TestNditer(t *testing.T) {
	shp := Shape{2, 3, 4, 5}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	fmt.Println("x: ", x, x.strides)

	xit := newnditer(x)
	for k, inc := 0, xit.Inc(); !xit.Done(); {
		v, n := xit.Next()
		for i := range v[:n] {
			i *= inc
			if float64(k) != v[i] {
				t.Logf("test failed. got: %v, exp: %v\n", v[i], k)
				t.Fail()
			}
			k++
		}
	}
	xit.Reset()

	y := permute(x, []int{0, 1, 3, 2})
	fmt.Println("y: ", y)
	exp := Reshape(Arange(0, float64(x.size)), x.shape).(*ndarray)
	ep := permute(exp, []int{0, 1, 3, 2})
	fmt.Println("ep: ", ep)

	yit := newnditer(y)
	eit := newnditer(ep)
	for k, inc := 0, yit.Inc(); !yit.Done(); {
		yv, ev, n := yit.ZipNext(eit)
		for i := range yv[:n] {
			i *= inc
			if ev[i] != yv[i] {
				t.Logf("test failed. got: %v, exp: %v\n", yv[i], ev[i])
				t.Fail()
			}
			k++
		}
	}
	yit.Reset()

	yv := y.View(Index{1, 0, 1, 1}, Shape{1, 2, 4, 3}).(*ndarray)
	fmt.Println("yv: ", yv)
	ep = y.View(Index{1, 0, 1, 1}, yv.shape).(*ndarray)
	fmt.Println("ep: ", ep)
	yit = newnditer(yv)
	eit = newnditer(ep)
	for k, inc := 0, yit.Inc(); !yit.Done(); {
		yv, ev, n := yit.ZipNext(eit)
		for i := range yv[:n] {
			i *= inc
			if ev[i] != yv[i] {
				t.Logf("test failed. got: %v, exp: %v\n", yv[i], ev[i])
				t.Fail()
			}
			k++
		}
	}
	yit.Reset()

	yvt := permute(yv, []int{2, 1, 0, 3})
	fmt.Println("yvt: ", yvt)
	ep = permute(yv, []int{2, 1, 0, 3})
	fmt.Println("ep: ", ep)
	yit = newnditer(yvt)
	eit = newnditer(ep)
	for k, inc := 0, yit.Inc(); !yit.Done(); {
		yv, ev, n := yit.ZipNext(eit)
		for i := range yv[:n] {
			i *= inc
			if ev[i] != yv[i] {
				t.Logf("test failed. got: %v, exp: %v\n", yv[i], ev[i])
				t.Fail()
			}
			k++
		}
	}
	yit.Reset()

	s := yit.Fold(40, func(f float64) float64 {
		return f * 2
	})

	if s != 4e3 {
		t.Logf("test failed. got: %v, exp: %v\n", s, 4e3)
		t.Fail()
	}

	nv := make([]float64, ComputeSize(shp))
	for i, k := 0, len(nv)-1; k >= 0; k, i = k-1, i+1 {
		nv[i] = float64(k)
	}
	n := New(shp, nv).(*ndarray)
	m := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	mit := newnditer(m)
	nit := newnditer(n)
	got := dot(mit, nit)
	if exp := 280840.0; got != exp {
		t.Logf("test failed. got: %v, exp: %v\n", got, exp)
		t.Fail()
	}
}

func ExampleNditer() {
	shp := Shape{3, 4, 5}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	it := newnditer(x)
	// it is an iterator into the array x.
	// It has four methods of interest - Done, Inc, Next and Reset.
	// Inc() returns the step we have to use to access individual
	// elements in the slice returned by Next.
	// Next returns a slice containing the row at each iteration
	// of the outer loop.
	// After we are Done(), we call Reset(), to put the iterator
	// back to its default state.
	for inc := it.Inc(); !it.Done(); {
		x, n := it.Next()
		for i := range x[:n] {
			i *= inc
			fmt.Println(x[i])
		}
	}
	it.Reset()
}

func dot(x, y *nditer) (s float64) {
	if !IsShapeSame(x.arr, y.arr) {
		panic("iterators must have same shape")
	}

	if x == y {
		if !x.isplain {
			for inc := x.Inc(); !x.Done(); {
				v, n := x.Next()
				for i := range v[:n] {
					i *= inc
					s += v[i] * v[i]
				}
			}
		} else {
			for !x.Done() {
				v, n := x.Next()
				for _, e := range v[:n] {
					s += e * e
				}
			}
		}
		x.Reset()
		return
	}

	inc := x.Inc()
	if !x.isplain {
		for !x.Done() {
			xv, yv, n := x.ZipNext(y)
			for i := range xv[:n] {
				i *= inc
				s += xv[i] * yv[i]
			}
		}
	} else {
		for !x.Done() {
			xv, yv, n := x.ZipNext(y)
			for i, v := range xv[:n] {
				s += yv[i] * v
			}
		}
	}
	x.Reset()
	y.Reset()
	return
}

func BenchmarkNditer(b *testing.B) {
	b.ReportAllocs()
	x := Rand(TestArrayShape).(*ndarray)
	y := Rand(TestArrayShape).(*ndarray)
	xit := newnditer(x)
	yit := newnditer(y)
	// inc := xit.Inc()
	a := rand.Float64()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// for xit.Reset(); !xit.Done(); {
		// 	v, n := xit.Next()
		// 	for i := range v[:n] {
		// 		i *= inc
		// 		v[i] = a
		// 	}
		// }
		a = dot(xit, yit)
	}
	_ = a * a
}

func BenchmarkNditerInd(b *testing.B) {
	b.ReportAllocs()
	x := Rand(TestArrayShape).(*ndarray)
	xit := newnditer(x)
	j := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		j = xit.ind(3)
	}
	_ = j * j
}

func TestFoo(t *testing.T) {
	b := Index{0, 0, 0, 0}
	shp := Shape{3, 2, 5, 4}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	fmt.Println("x: ", x)
	b = Index{1, 0, 2, 1}
	xv := x.View(b, Shape{2, 2, 2, 3}).(*ndarray)
	fmt.Println("xv: ", xv)

	it2d := func(shp, str Shape, x []float64) {
		stp0, stp1 := str[0], str[1]
		if stp1 > 1 {
			if stp1 > stp0 {
				for k := 0; k < shp[0]; k++ {
					b := k * stp0
					for j := 0; j < shp[1]; j++ {
						l := j * stp1
						fmt.Printf("k: %v, j: %v\n", k, j)
						fmt.Printf("data[b:%d]: %v\n", b+l, x[b+l:][:shp[1]-1])
					}
				}
			} else {
				for k := 0; k < shp[0]; k++ {
					b := k * stp1
					for j := 0; j < shp[1]; j++ {
						l := j * stp0
						fmt.Printf("k: %v, j: %v\n", k, j)
						fmt.Printf("data[b:%d]: %v\n", b+l, x[b+l:][:shp[1]-1])
					}
				}
			}
			return
		}
		for k := 0; k < shp[0]; k++ {
			b := k * stp0
			fmt.Printf("k: %v, b: %v\n", k, b)
			fmt.Printf("data[b:%d]: %v\n", b, x[b:][:shp[1]])
		}
	}

	iterate := func(xv *ndarray) {
		// goal is to iterate over the view without too much overhead
		if xv.ndims < 3 {
			it2d(xv.shape, xv.strides, xv.data)
			return
		}

		nd := xv.ndims
		shp := xv.shape
		str := xv.strides

		shape := make(Shape, nd)
		shpnd := shp[:nd-2]
		copy(shape, shpnd)
		for i := nd - 2; i < nd; i++ {
			shape[i] = 1
		}

		shp2d := shp[nd-2:]
		str2d := str[nd-2:]
		strnd := str[:nd-2]

		istr := ComputeStrides(shpnd)
		ind := make(Index, nd-2)
		for n := 0; n < ComputeSize(shpnd); n++ {
			Ind2sub(istr, n, ind)
			j := Sub2ind(strnd, ind)
			it2d(shp2d, str2d, xv.data[j:])
		}
	}

	// iterate(x)
	// iterate(xv)
	y := permute(xv, []int{2, 3, 1, 0})
	fmt.Println("y: ", y, y.shape)
	iterate(y)
	// desired usage ->
	// for it := newiternd(xv); !it.done(); {
	// 	for _, v := range it.next() {
	// 		fmt.Println(v)
	// 	}
	// }

	// iterate(y)
	// TODO: iterator should return a different inc based on ndims of the array being iterated on
	// TODO: implement an iterator that advances the index by 1 position
	/* 	type cntr struct {
	   		step0, step1, step2 int
	   		// len0, len1, len2    int
	   		str  [3]int
	   		istr [3]float64
	   		// sub  [3]int
	   		size     int
	   		k, j     int
	   		plaininc bool
	   		data     []float64
	   	}

	   	c := cntr{
	   		step0: xv.strides[xv.ndims-3], step1: xv.strides[xv.ndims-2], step2: xv.strides[xv.ndims-1],
	   		size: ComputeSize(shpnd) * shp2d[0] * shp2d[1],
	   		// len0: ComputeSize(shpnd), len1: shp2d[0], len2: shp2d[1],
	   		str:      [3]int{shp2d[1] * shp2d[0], shp2d[1], 1},
	   		plaininc: len(xv.data) == xv.size,
	   		data:     xv.data,
	   	}
	   	// c.str = [3]int{c.len2 * c.len1, c.len2, 1}
	   	c.istr = [3]float64{1.0 / float64(c.str[0]), 1.0 / float64(c.str[1]), 1.0 / float64(c.str[2])}
	   	// c.size = c.len0 * c.len1 * c.len2

	   	sub := func(c *cntr, k int) (s int) {
	   		j := 0
	   		j = int(float64(k) * c.istr[0])
	   		s += j * c.step0
	   		k -= j * c.str[0]
	   		j = int(float64(k) * c.istr[1])
	   		s += j * c.step1
	   		k -= j * c.str[1]
	   		j = int(float64(k) * c.istr[2])
	   		s += j * c.step2
	   		k -= j * c.str[2]
	   		return
	   	}

	   	inc := func(c *cntr) {
	   		if c.plaininc {
	   			c.k++
	   			c.j = c.k
	   			return
	   		}

	   		c.k++
	   		c.j = sub(c, c.k)
	   	}

	   	at := func(c *cntr) float64 {
	   		return c.data[c.j]
	   	}

	   	done := func(c *cntr) bool {
	   		return c.k == c.size
	   	}

	   	reset := func(c *cntr) {
	   		c.k = 0
	   	}
	   	fmt.Println("c.step: ", c.step0, c.step1, c.step2, c.size)
	   	for reset(&c); !done(&c); inc(&c) {
	   		fmt.Printf("c.k: %d, c.j: %d, at: %.4f\n", c.k, c.j, at(&c))
	   	}

	   	const N = 1e6
	   	v := 0.0

	   	st := time.Now()
	   	for i := 0; i < N; i++ {
	   		for reset(&c); !done(&c); inc(&c) {
	   			v = at(&c)
	   		}
	   	}
	   	el := time.Since(st) / N
	   	_ = v * v
	   	fmt.Println("time per op: ", el)
	*/
}

func BenchmarkCounter(b *testing.B) {
	b.ReportAllocs()
	// x := Rand(TestArrayShape)
	c := newcounter(Index{0, 0, 0}, TestArrayShape)
	// ind := Index{0, 0, 0}
	j := 0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for c.reset(); !c.done(); c.inc() {
			// ind = c.at()
			// ind = c.i
			j = c.k
		}
	}
	_ = j * j
	// _ = len(ind)
}

func BenchmarkIterNext(b *testing.B) {
	b.ReportAllocs()
	x := Rand(TestArrayShape).(*ndarray)
	isub := Index{0, 0, 0}
	j := Index{0, 0, 0}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for k := 0; k < x.size; k++ {
			j = x.ind2sub(k, isub)
		}
	}
	_ = len(j)
}

type cntr struct {
	step0, step1, step2 int
	str                 [3]int
	istr                [3]float64
	size                int
	k, j                int
	plaininc            bool
	data                []float64
}

func newcntr(array *ndarray) *cntr {
	n := array.ndims
	str := array.strides
	shpnd := array.shape[:n-2]
	shp2d := array.shape[n-2:]

	c := &cntr{
		step0:    str[n-3],
		step1:    str[n-2],
		step2:    str[n-1],
		size:     ComputeSize(shpnd) * shp2d[0] * shp2d[1],
		str:      [3]int{shp2d[1] * shp2d[0], shp2d[1], 1},
		plaininc: len(array.data) == array.size,
		data:     array.data,
	}
	c.istr = [3]float64{1.0 / float64(c.str[0]), 1.0 / float64(c.str[1]), 1.0 / float64(c.str[2])}
	return c
}

func (c *cntr) inc() {
	if c.plaininc {
		c.j = c.k
		c.k++
		return
	}

	c.k++
	c.j = c.sub(c.k)
}

func (c *cntr) at() float64 { return c.data[c.j] }

func (c *cntr) done() bool { return c.k == c.size }

func (c *cntr) reset() { c.k = 0 }

func (c *cntr) sub(k int) (s int) {
	j := 0
	j = int(float64(k) * c.istr[0])
	s += j * c.step0
	k -= j * c.str[0]
	j = int(float64(k) * c.istr[1])
	s += j * c.step1
	k -= j * c.str[1]
	j = int(float64(k) * c.istr[2])
	s += j * c.step2
	k -= j * c.str[2]
	return
}

func TestCntr(t *testing.T) {
	b := Index{0, 0, 0, 0}
	shp := Shape{3, 2, 5, 2}
	x := Reshape(Arange(0, 60), shp).(*ndarray)
	fmt.Println("x: ", x)
	b = Index{1, 0, 2, 1}
	xv := x.View(b, Shape{2, 3, 1}).(*ndarray)
	fmt.Println("xv: ", xv)

	exp := []float64{25, 27, 29, 35, 37, 39}
	c := newcntr(xv)
	for _, k := range exp {
		v := c.at()
		if v != k {
			t.Logf("test failed. got: %v, exp: %v\n", v, k)
			t.Fail()
		}
		fmt.Printf("got: %4.4f, exp: %4.4f\n", v, k)
		c.inc()
	}
}

func fill(x *cntr, a float64) {
	if x.plaininc {
		for j := range x.data {
			x.data[j] = a
		}
		return
	}

	for ; !x.done(); x.inc() {
		x.data[x.j] = a
	}
	x.reset()
}

func BenchmarkCntr(b *testing.B) {
	b.ReportAllocs()
	x := Rand(TestArrayShape).(*ndarray)
	xv := x.View(Index{33, 33, 33}, Shape{33, 33, 33}).(*ndarray)
	c := newcntr(xv)
	// v := 0.0
	v := rand.Float64()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		fill(c, v)
	}
	_ = v * v
}
