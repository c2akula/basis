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
	shp, str := array.shape, array.strides
	shpnd := shp[:nd-1]
	shp2d := shp[nd-2:]
	str2d := str[nd-2:]

	it := &nditer{
		arr:     array,
		istr:    make([]float64, nd),
		sub:     make(Index, nd), // holds the subscript version of the current position
		len:     ComputeSize(shpnd),
		isplain: !(str2d[1] != 1),
	}
	copy(it.shp2d[:], shp2d)
	copy(it.str2d[:], str2d)

	// store the inverse of the strides for faster ind2sub calculation
	ishp := make(Shape, nd)
	copy(ishp, shp)
	for i := nd - 1; i < nd; i++ {
		ishp[i] = 1
	}
	it.str = ComputeStrides(ishp)
	// it.str = array.strides
	for i, s := range it.str {
		it.istr[i] = 1 / float64(s)
	}
	fmt.Printf("it.str: %v, it.istr: %v\n", it.str, it.istr)
	return it
}

func (it *nditer) ind(k int) (s int) {
	for i, n := range it.istr[:it.arr.ndims-1] {
		j := int(float64(k) * n)
		s += j * it.arr.strides[i]
		k -= j * it.str[i]
		it.sub[i] = j
	}
	// for _, n := range it.str {
	// 	s += k * n
	// }
	// fmt.Println("s: ", s)
	// fmt.Println("it.sub: ", it.sub)
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
	fmt.Println("next: ", x)
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
	fmt.Println("x: ", x)

	xit := newnditer(x)
	for k, inc := 0, xit.Inc(); !xit.Done(); {
		v, n := xit.Next()
		for i := range v[:n] {
			i *= inc
			fmt.Println("i: ", i, n)
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

	/* 	s := yit.Fold(40, func(f float64) float64 {
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
	*/
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

/*
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
*/
