package main

import (
	"bytes"
	"fmt"
	"github.com/nfnt/resize"
	tensorflow "github.com/wamuir/graft/tensorflow"
	"image"
	"image/jpeg"
	"log"
	"net/http"
)

// Предобработка изображения
func preprocessImage(img image.Image) (*tensorflow.Tensor, error) {
	resized := resize.Resize(180, 180, img, resize.Lanczos3)
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, resized, nil); err != nil {
		return nil, fmt.Errorf("failed to encode image: %v", err)
	}

	imageBytes := buf.Bytes()
	var imageData [180][180][3]float32
	index := 0
	for i := range imageData {
		for j := range imageData[i] {
			for k := range imageData[i][j] {
				if index < len(imageBytes) {
					imageData[i][j][k] = float32(imageBytes[index]) / 255.0
					index++
				}
			}
		}
	}

	tensor, err := tensorflow.NewTensor([1][180][180][3]float32{imageData})
	if err != nil {
		return nil, fmt.Errorf("failed to create tensor: %v", err)
	}

	return tensor, nil
}

// Загрузка модели
func loadModel(modelDir string) (*tensorflow.SavedModel, error) {
	model, err := tensorflow.LoadSavedModel(modelDir, []string{"serve"}, nil)
	if err != nil {
		return nil, fmt.Errorf("could not load model: %v", err)
	}
	return model, nil
}

// Веб-эндпоинт предсказания
func predict(w http.ResponseWriter, r *http.Request) {
	file, _, err := r.FormFile("imagefile")
	if err != nil {
		http.Error(w, "could not read image", http.StatusBadRequest)
		return
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		http.Error(w, "invalid image format", http.StatusBadRequest)
		return
	}

	tensor, err := preprocessImage(img)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	model, err := loadModel("modelDir")
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer model.Session.Close()

	// Выбор операции на основе сигнатуры
	sigChoice := r.URL.Query().Get("sig")
	if sigChoice == "" {
		sigChoice = "serving_default"
	}

	// Создание карты для input операций
	inputs := map[tensorflow.Output]*tensorflow.Tensor{
		model.Graph.Operation("serving_default_input_tensor").Output(0): tensor,
	}

	// Определяем output операции в зависимости от сигнатуры
	outputOp := model.Graph.Operation("StatefulPartitionedCall").Output(0)
	if outputOp == (tensorflow.Output{}) {
		http.Error(w, "Invalid or missing output operation", http.StatusInternalServerError)
		return
	}

	// Выполняем предсказание
	results, err := model.Session.Run(
		inputs,
		[]tensorflow.Output{outputOp},
		nil,
	)
	if err != nil {
		http.Error(w, fmt.Sprintf("could not run the session: %v", err), http.StatusInternalServerError)
		return
	}

	// Обработка результата
	probability := results[0].Value().([][]float32)[0][0]
	prediction := "Скорее всего, это кот"
	if probability > 0.5 {
		prediction = "Кажется, это собака"
	}

	fmt.Fprintf(w, "Prediction: %s", prediction)
}

func main() {
	http.HandleFunc("/predict", predict)
	log.Fatal(http.ListenAndServe(":8080", nil))
}
