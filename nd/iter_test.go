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

	for k := 0; k < res.ndims; k++ {
		shp := res.shape
		str := res.strides
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
	fmt.Printf("res.shp: %v, res.str: %v\n", res.shape, res.strides)
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
				return
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

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		// it = fill(it, a)
		a = dot(xit, yit)
	}
	// _ = it.arr[0] * it.arr[0]
	_ = a * a
}

func TestFoo(t *testing.T) {
	b := Index{0, 0, 0, 0}
	shp := Shape{3, 2, 5, 4}
	x := Reshape(Arange(0, float64(ComputeSize(shp))), shp).(*ndarray)
	fmt.Println("x: ", x)
	b = Index{1, 0, 2, 1}
	xv := x.View(b, Shape{2, 2, 3}).(*ndarray)
	// xv := x.View(b, shp).(*ndarray)
	fmt.Println("xv: ", xv)

	// y := transpose(xv)
	// fmt.Println("y: ", y)
	it2d := func(shp, str Shape, x []float64) {
		stp0, stp1 := str[0], str[1]
		if stp1 > 1 {
			for k := 0; k < shp[1]; k++ {
				b := k * stp1
				fmt.Printf("k: %v, b: %v\n", k, b)
				fmt.Printf("data[b:%d]: %v\n", b, x[b:b+shp[0]])
			}
			return
		}
		for k := 0; k < shp[0]; k++ {
			b := k * stp0
			fmt.Printf("k: %v, b: %v\n", k, b)
			fmt.Printf("data[b:%d]: %v\n", b, x[b:b+shp[1]])
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
		shpnd := shp[:xv.ndims-2]
		// strnd := xv.strides[:xv.ndims-2]
		shp2d := shp[xv.ndims-2:]
		// str2d := xv.strides[xv.ndims-2:]
		// step := strnd[xv.ndims-3]
		// fmt.Println("xv.data: ", xv.data)
		// fmt.Printf("shpnd: %v, strnd: %v, shp2d: %v, str2d: %v\n", shpnd, strnd, shp2d, str2d)

		step0 := str[nd-3]
		step1 := str[nd-2]
		step2 := str[nd-1]

		for n := 0; n < ComputeSize(shpnd); n++ {
			b := n * step0
			// it2d(shp2d, str2d, xv.data[b:b+ComputeSize(shp2d)])
			if step2 != 1 {
				for k := 0; k < shp2d[1]; k++ {
					l := b + k*step2
					fmt.Printf("nd: data[b:%d]=%v\n", b, xv.data[l:l+shp2d[0]])
				}
			} else {
				for k := 0; k < shp2d[0]; k++ {
					l := b + k*step1
					fmt.Printf("nd: data[b:%d]=%v\n", b, xv.data[l:l+shp2d[1]])
				}
			}
		}
	}

	// iterate(x)
	// iterate(xv)
	y := permute(xv, []int{0, 2, 1})
	fmt.Println("y: ", y)
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
