package nd

import (
	"fmt"
	"testing"
)

func TestPrint(t *testing.T) {
	shp := Shape{3, 4, 5, 6}
	x := Arange(0, float64(ComputeSize(shp))).Reshape(shp)
	fmt.Println(x.String())
}
