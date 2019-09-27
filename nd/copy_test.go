package nd

import (
	"fmt"
	"testing"
)

func TestCopy(t *testing.T) {
	shp := Shape{3, 4, 5}
	// Copy plain
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	fmt.Println("x: ", x)
	y := Zeroslike(x)
	Copy(y.Iter(), x.Iter())
	fmt.Println("y: ", y)

	// Copy views
	xv := x.View(Index{1, 0, 1}, Shape{2, 3, 2})
	fmt.Println("xv: ", xv)
	yv := y.View(Index{0, 1, 2}, xv.shape)
	fmt.Println("yv: ", yv)
	Copy(yv.Iter(), xv.Iter())
	fmt.Println("y*: ", y)

	// Copy Broadcast
	m := Zeros(Shape{3, 4, 1})
	if err := Broadcast(m, x); err != nil {
		panic(err)
		t.Fail()
	}
	Copy(m.Iter(), x.Iter())
	fmt.Println("m: ", m)
}
