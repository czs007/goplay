package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	milvusClient "github.com/xiaocai2333/milvus-sdk-go/v2/client"
	"github.com/xiaocai2333/milvus-sdk-go/v2/entity"
	"google.golang.org/grpc"
	"io"
	"math"
	"os"
	"path"
	"strconv"
	"time"
)

const (
	CollectionName       = "taip"
	DefaultPartitionName = "_default"
	Dim                  = 768
	QueryFile = "query.npy"
	DataPath = "/data/milvus/raw_data/zjlab"
	RunTime = 1000
	VecFieldName = "vec"
)

func createClient(addr string) milvusClient.Client{
	opts := []grpc.DialOption{grpc.WithInsecure(),
		grpc.WithBlock(),                //block connect until healthy or timeout
		grpc.WithTimeout(20*time.Second)} // set connect timeout to 2 Second
	client, err := milvusClient.NewGrpcClient(context.Background(), addr, opts...)
	if err != nil {
		panic(err)
	}
	return client
}

func BytesToFloat32(bits []byte) []float32 {
	vectors := make([]float32, 0)
	start, end := 0, 4
	for start < len(bits) && end <= len(bits) {
		num := math.Float32frombits(binary.LittleEndian.Uint32(bits[start:end]))
		vectors = append(vectors, num)
		start += 4
		end += 4
	}
	return vectors
}

func ReadBytesFromFile(nq int, filePath string) []byte {
	f, err := os.Open(filePath)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	r := bufio.NewReader(f)
	chunks := make([]byte, 0)
	buf := make([]byte, 4*Dim)
	readByte := 0
	for readByte < nq*Dim*4 {
		n, err := r.Read(buf)
		if err != nil && err != io.EOF {
			panic(err)
		}
		if 0 == n {
			break
		}
		chunks = append(chunks, buf[:n]...)
		readByte += n
	}
	//fmt.Println(len(chunks))
	return chunks[:nq*Dim*4]
}

func generatedEntities(nq int) []entity.Vector {
	filePath := path.Join(DataPath, QueryFile)
	bits := ReadBytesFromFile(nq, filePath)
	vectors := make([]entity.Vector, 0)
	for i := 0; i < nq; i++ {
		var vector entity.FloatVector = BytesToFloat32(bits[i*Dim*4:(i+1)*Dim*4])
		//fmt.Println(len(vector))
		vectors = append(vectors, vector)
	}
	return vectors
}

func generateInsertFile(x int) string {
	return "binary_" + strconv.Itoa(x) +"d_" + fmt.Sprintf("%5d", x) + ".npy"
}

func generateInsertPath(x int) string {
	return path.Join(DataPath, generateInsertFile(x))
}

func generateInertData(step int, t string) entity.Column {
	var colData entity.Column
	switch t {
	case "Int64":
		intData := make([]int64, step)
		for i := ID; i < PER_FILE_ROWS; i++ {
			intData[i] = int64(i)
		}
		colData = entity.NewColumnInt64("id", intData)
	case "FloatVector":
		colData = entity.NewColumnFloatVector(VecFieldName, Dim, generatedInsertEntities(step))
	default:
		panic(fmt.Sprintf("column type %s is not supported", t))
	}
	return colData
}

func generatedInsertEntities(num int) [][]float32 {
	filePath := generateInsertPath(num)
	bits := ReadBytesFromFile(PER_FILE_ROWS, filePath)
	vectors := make([][]float32, 0)
	for i := 0; i < PER_FILE_ROWS; i++ {
		vector := BytesToFloat32(bits[i*Dim*4:(i+1)*Dim*4])
		//fmt.Println(len(vector))
		vectors = append(vectors, vector)
	}
	return vectors
}

func main() {
	var addr = *flag.String("host", "172.18.50.4:19530", "milvus addr")
	var dataset = *flag.String("dataset", "taip", "dataset for test")
	var indexType = *flag.String("index", "HNSW", "index type for collection, HNSW | IVF_FLAT")
	var process = *flag.Int("process", 1, "goroutines for test")
	var operation = *flag.String("op", "Search", "what do you want to do")

	fmt.Printf("host: %s, dataset: %s, operation: %s, index type: %s, process: %d \n", addr, dataset,
		operation, indexType, process)

	client := createClient(addr)
	defer client.Close()
	if operation == "Insert" {
		//TODO: Create collection and insert data
	}
	if operation == "Search" {
		Search(client, dataset, indexType, process)
	}
	if operation == "Index" {
		// TODO: Create index
	}
	if operation == "Load" {
		// TODO: Load collection
	}
}
