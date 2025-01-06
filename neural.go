// Copyright 2024 The TXT Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/pointlander/gradient/tf32"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Neural is a neural network
type Neural struct {
	Set    tf32.Set
	Others tf32.Set
	L1     tf32.Meta
	L2     tf32.Meta
	L3     tf32.Meta
	L4     tf32.Meta
	Loss   tf32.Meta
}

// Load loads a neural network from a file
func Load() (networks [256]Neural) {
	set := tf32.NewSet()
	cost, epochs, err := set.Open("set.db")
	if err != nil {
		panic(err)
	}
	fmt.Println(cost, epochs)

	for i := range networks {
		others := tf32.NewSet()
		others.Add("input", 256)
		others.Add("output", 256)

		for i := range others.Weights {
			w := others.Weights[i]
			w.X = w.X[:cap(w.X)]
		}

		l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("w1_%d", i)), others.Get("input")), set.Get(fmt.Sprintf("b1_%d", i))))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("w2_%d", i)), l1), set.Get(fmt.Sprintf("b2_%d", i))))
		l3 := tf32.Everett(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("w3_%d", i)), l2), set.Get(fmt.Sprintf("b3_%d", i))))
		l4 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("w4_%d", i)), l3), set.Get(fmt.Sprintf("b4_%d", i))))
		loss := tf32.Quadratic(l4, others.Get("output"))

		/*sumRows := tf64.U(SumRows)

		query := tf64.Mul(set.Get("query"), others.Get("input"))
		key := tf64.Mul(set.Get("key"), others.Get("input"))
		value := tf64.Mul(set.Get("value"), others.Get("input"))
		l1 := tf64.Add(others.Get("input"), tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(query, key)), tf64.T(value))))
		l2 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w1"), l1), set.Get("b1")))
		l3 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), l2), set.Get("b2")))
		query2 := tf64.Mul(set.Get("query2"), l3)
		key2 := tf64.Mul(set.Get("key2"), l3)
		value2 := tf64.Mul(set.Get("value2"), l3)
		l4 := tf64.Add(l3, tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(query2, key2)), tf64.T(value2))))
		l5 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w3"), l4), set.Get("b3")))
		l6 := tf64.Sigmoid(sumRows(tf64.Add(tf64.Mul(set.Get("w4"), l5), set.Get("b4"))))
		loss := tf64.Quadratic(l6, others.Get("output"))*/

		networks[i] = Neural{
			Set:    set,
			Others: others,
			L1:     l1,
			L2:     l2,
			L3:     l3,
			L4:     l4,
			//L5:     l5,
			//L6:     l6,
			Loss: loss,
		}
	}

	return networks
}

// SumRows sums the rows of the matrix
func SumRows(k tf32.Continuation, node int, a *tf32.V, options ...map[string]interface{}) bool {
	size, width := len(a.X), a.S[0]
	total := float32(0.0)
	c := tf32.NewV(width)
	c.X = c.X[:cap(c.X)]
	for i := 0; i < size; i += width {
		for j, ax := range a.X[i : i+width] {
			c.X[j] += ax
		}
		total++
	}
	for i := range c.X {
		c.X[i] /= total
	}
	if k(&c) {
		return true
	}
	for i := 0; i < size; i += width {
		for j := range a.D[i : i+width] {
			a.D[i+j] += c.D[j] / total
		}
	}
	return false
}

// Learn learn a neural network
func Learn(data []byte) Neural {
	rng := rand.New(rand.NewSource(1))
	set := tf32.NewSet()
	for i := 0; i < 256; i++ {
		/*set.Add("query", 256, 256)
		set.Add("key", 256, 256)
		set.Add("value", 256, 256)
		set.Add("query2", 256, 256)
		set.Add("key2", 256, 256)
		set.Add("value2", 256, 256)
		set.Add("w1", 256, 256)
		set.Add("b1", 256)
		set.Add("w2", 512, 256)
		set.Add("b2", 256)
		set.Add("w3", 256, 256)
		set.Add("b3", 256)
		set.Add("w4", 512, 256)
		set.Add("b4", 256)*/
		set.Add(fmt.Sprintf("w1_%d", i), 256, 256)
		set.Add(fmt.Sprintf("b1_%d", i), 256)
		set.Add(fmt.Sprintf("w2_%d", i), 512, 256)
		set.Add(fmt.Sprintf("b2_%d", i), 256)
		set.Add(fmt.Sprintf("w3_%d", i), 512, 256)
		set.Add(fmt.Sprintf("b3_%d", i), 256)
		set.Add(fmt.Sprintf("w4_%d", i), 512, 256)
		set.Add(fmt.Sprintf("b4_%d", i), 256)
	}

	for i := range set.Weights {
		w := set.Weights[i]
		if strings.HasPrefix(w.N, "b") {
			w.X = w.X[:cap(w.X)]
			w.States = make([][]float32, StateTotal)
			for i := range w.States {
				w.States[i] = make([]float32, len(w.X))
			}
			continue
		}
		factor := math.Sqrt(2.0 / float64(w.S[0]))
		for i := 0; i < cap(w.X); i++ {
			w.X = append(w.X, float32(rng.NormFloat64()*factor))
		}
		w.States = make([][]float32, StateTotal)
		for i := range w.States {
			w.States[i] = make([]float32, len(w.X))
		}
	}

	type Item struct {
		Vector [256]float32
		Symbol byte
	}
	inputs := [256]chan *Item{}
	done := make(chan bool, 8)

	process := func(input chan *Item, prefix byte) {
		others := tf32.NewSet()
		//others.Add("input", 256, Size)
		others.Add("input", 256)
		others.Add("output", 256)

		for i := range others.Weights {
			w := others.Weights[i]
			w.X = w.X[:cap(w.X)]
		}

		l1 := tf32.Everett(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("w1_%d", prefix)), others.Get("input")), set.Get(fmt.Sprintf("b1_%d", prefix))))
		l2 := tf32.Everett(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("w2_%d", prefix)), l1), set.Get(fmt.Sprintf("b2_%d", prefix))))
		l3 := tf32.Everett(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("w3_%d", prefix)), l2), set.Get(fmt.Sprintf("b3_%d", prefix))))
		l4 := tf32.Sigmoid(tf32.Add(tf32.Mul(set.Get(fmt.Sprintf("w4_%d", prefix)), l3), set.Get(fmt.Sprintf("b4_%d", prefix))))
		loss := tf32.Quadratic(l4, others.Get("output"))

		/*sumRows := tf64.U(SumRows)

		options := map[string]interface{}{
			"rng": rng,
		}
		query := tf64.Mul(set.Get("query"), others.Get("input"))
		key := tf64.Mul(set.Get("key"), others.Get("input"))
		value := tf64.Mul(set.Get("value"), others.Get("input"))
		l1 := tf64.Add(others.Get("input"), tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(query, key)), tf64.T(value))))
		l2 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w1"), tf64.Dropout(l1, options)), set.Get("b1")))
		l3 := tf64.Sigmoid(tf64.Add(tf64.Mul(set.Get("w2"), tf64.Dropout(l2, options)), set.Get("b2")))
		query2 := tf64.Mul(set.Get("query2"), l3)
		key2 := tf64.Mul(set.Get("key2"), l3)
		value2 := tf64.Mul(set.Get("value2"), l3)
		l4 := tf64.Add(l3, tf64.T(tf64.Mul(tf64.Softmax(tf64.Mul(query2, key2)), tf64.T(value2))))
		l5 := tf64.Everett(tf64.Add(tf64.Mul(set.Get("w3"), tf64.Dropout(l4, options)), set.Get("b3")))
		l6 := tf64.Sigmoid(sumRows(tf64.Add(tf64.Mul(set.Get("w4"), tf64.Dropout(l5, options)), set.Get("b4"))))
		loss := tf64.Quadratic(l6, others.Get("output"))*/

		points := make(plotter.XYs, 0, 8)
		fmt.Println("learning:", len(data))
		i := 0
		for in := range input {
			pow := func(x float32) float32 {
				y := math.Pow(float64(x), float64(i+1))
				if math.IsNaN(y) || math.IsInf(y, 0) {
					return 0
				}
				return float32(y)
			}

			others.Zero()
			Softmax(in.Vector[:], 1.0)
			input := others.ByName["input"].X
			for j := range input {
				input[j] = in.Vector[j]
			}
			output := others.ByName["output"].X
			for j := range output {
				output[j] = .001
			}
			output[in.Symbol] = 1

			set.Zero()
			cost := tf32.Gradient(loss).X[0]
			if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
				break
			}

			norm := float32(0.0)
			for _, p := range set.Weights {
				if !strings.HasSuffix(p.N, fmt.Sprintf("_%d", prefix)) {
					continue
				}
				for _, d := range p.D {
					norm += d * d
				}
			}
			norm = float32(math.Sqrt(float64(norm)))
			b1, b2 := pow(B1), pow(B2)
			scaling := float32(1.0)
			if norm > 1 {
				scaling = 1 / norm
			}
			for _, w := range set.Weights {
				if !strings.HasSuffix(w.N, fmt.Sprintf("_%d", prefix)) {
					continue
				}
				for l, d := range w.D {
					g := d * scaling
					m := B1*w.States[StateM][l] + (1-B1)*g
					v := B2*w.States[StateV][l] + (1-B2)*g*g
					w.States[StateM][l] = m
					w.States[StateV][l] = v
					mhat := m / (1 - b1)
					vhat := v / (1 - b2)
					if vhat < 0 {
						vhat = 0
					}
					w.X[l] -= Eta * mhat / (float32(math.Sqrt(float64(vhat))) + 1e-8)
				}
			}
			points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
			if i%1024 == 0 {
				fmt.Println(i, cost)
			}
			i++
		}

		p := plot.New()

		p.Title.Text = "epochs vs cost"
		p.X.Label.Text = "epochs"
		p.Y.Label.Text = "cost"

		scatter, err := plotter.NewScatter(points)
		if err != nil {
			panic(err)
		}
		scatter.GlyphStyle.Radius = vg.Length(1)
		scatter.GlyphStyle.Shape = draw.CircleGlyph{}
		p.Add(scatter)

		err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("epochs/epochs_%d.png", prefix))
		if err != nil {
			panic(err)
		}

		done <- true
	}

	for i := range inputs {
		inputs[i] = make(chan *Item, 8)
		go process(inputs[i], byte(i))
	}

	for i := 0; i < 3*len(data); i++ {
		index := rng.Intn(len(data) - 256)
		m := NewMixer()
		end := index + 8 + rng.Intn(120)
		for j := index; j < end; j++ {
			m.Add(data[j])
		}
		item := Item{
			Vector: m.MixFloat32(), //m.Raw()
			Symbol: data[end],
		}
		inputs[m.Markov[0]] <- &item
	}

	for i := range inputs {
		close(inputs[i])
		fmt.Println("close", i)
	}

	for i := 0; i < 256; i++ {
		<-done
		fmt.Println("done", i)
	}

	err := set.Save("set.db", 0.0, 3*len(data))
	if err != nil {
		panic(err)
	}

	return Neural{}
}

// Inference performs inference of the neural network
func (n *Neural) Inference(input [256]float32) int {
	symbol, max := 0, float32(0.0)
	in := n.Others.ByName["input"].X
	for i := range in {
		in[i] = input[i]
	}
	n.L4(func(a *tf32.V) bool {
		for i, v := range a.X {
			if v > max {
				max, symbol = v, i
			}
		}
		return true
	})

	return symbol
}

// Distribution performs inference of the neural network
func (n *Neural) Distribution(input []float32) (d []float32) {
	in := n.Others.ByName["input"].X
	for i := range in {
		in[i] = input[i]
	}
	n.L4(func(a *tf32.V) bool {
		d = a.X
		return true
	})

	return d
}
