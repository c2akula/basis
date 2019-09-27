package nd

import (
	"math/rand"
	"testing"
)

func TestAxpy(t *testing.T) {
	t.Run("axpy_plain", func(t *testing.T) {
		shp := Shape{3, 4, 5}
		x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
		y := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
		exp := Zeroslike(y)

		eit := exp.Iter()

		for it := Zip(x, y); it.Next(); eit.Next() {
			xv, yv := it.Get()
			en, ev, es := eit.Get()
			for j, k, l := 0, 0, 0; en != 0; en-- {
				ev[l] = xv.Data[j] + yv.Data[k]
				j += xv.Step
				k += yv.Step
				l += es
			}
		}

		Axpy(1, x.Iter(), y.Iter())

		for it := Zip(exp, y); it.Next(); {
			ev, yv := it.Get()
			for j, k := 0, 0; ev.Size != 0; ev.Size-- {
				if ev.Data[j] != yv.Data[k] {
					t.Logf("test 'Axpy' failed. exp: %v\n, got: %v\n", exp, y)
					t.Fail()
				}
				j += ev.Step
				k += yv.Step
			}
		}
	})

	t.Run("axpy_view", func(t *testing.T) {
		x := Reshape(Arange(0, 60), Shape{3, 4, 5}).View(Index{1, 0, 1}, Shape{2, 3})
		y := Reshape(Arange(0, 60), Shape{3, 4, 5}).View(Index{1, 1, 1}, Shape{2, 3})
		exp := Zeroslike(y)

		eit := exp.Iter()

		for it := Zip(x, y); it.Next(); eit.Next() {
			xv, yv := it.Get()
			en, ev, es := eit.Get()
			for j, k, l := 0, 0, 0; en != 0; en-- {
				ev[l] = xv.Data[j] + yv.Data[k]
				j += xv.Step
				k += yv.Step
				l += es
			}
		}

		Axpy(1, x.Iter(), y.Iter())
		for it := Zip(exp, y); it.Next(); {
			ev, yv := it.Get()
			for j, k := 0, 0; ev.Size != 0; ev.Size-- {
				if ev.Data[j] != yv.Data[k] {
					t.Logf("test 'Axpy view' failed. exp: %v\n, got: %v\n", exp, y)
					t.Fail()
				}
				j += ev.Step
				k += yv.Step
			}
		}
	})
}

func BenchmarkAxpy(bn *testing.B) {
	bn.ReportAllocs()
	a := rand.Float64()
	TestArrayShape = Shape{1e2, 10}
	x := Rand(TestArrayShape).Iter()
	y := Rand(TestArrayShape).Iter()
	bn.ResetTimer()
	for i := 0; i < bn.N; i++ {
		Axpy(a, x, y)
	}
}
