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
	"strings"
	"time"
)

const (
	CollectionName       = "taip"
	DefaultPartitionName = "_default"
	RunTime      = 10000
	VecFieldName = "vec"

	TaipDataPath = "/data/milvus/raw_data/zjlab"
	SiftDataPath = "/data/milvus/raw_data/sift"
	QueryFile    = "query.npy"
)

var (
	Dim = 768
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

func generatedEntities(dataPath string, nq int) []entity.Vector {
	filePath := path.Join(dataPath, QueryFile)
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
	return "binary_" + strconv.Itoa(Dim) +"d_" + fmt.Sprintf("%05d", x) + ".npy"
}

func generateInsertPath(dataPath string, x int) string {
	return path.Join(dataPath, generateInsertFile(x))
}

type Strings []string

func newSliceValue(vals []string, p *[]string) *Strings {
	*p = vals
	return (*Strings)(p)
}

func (s *Strings) Set(val string) error {
	*s = Strings(strings.Split(val, ","))
	return nil
}

func (s *Strings) Get() interface{} {
	return []string(*s)
}

func (s *Strings) String() string {
	return strings.Join([]string(*s), ",")
}

var (
	addr       string
	dataset    string
	indexType  string
	process    int
	operation  string
	partitions []string
)

func init() {
	flag.StringVar(&addr, "host", "172.18.50.4:19530", "milvus addr")
	flag.StringVar(&dataset, "dataset", "taip", "dataset for test")
	flag.StringVar(&indexType, "index", "FLAT", "index type for collection, HNSW | IVF_FLAT | FLAT")
	flag.StringVar(&operation, "op", "", "what do you want to do")
	flag.Var(newSliceValue([]string{}, &partitions), "p", "partitions which you want to load")
	flag.IntVar(&process, "process", 1, "goroutines for test")
}

func main() {
	flag.Parse()
	fmt.Printf("host: %s, dataset: %s, operation: %s, index type: %s, process: %d, partitions: %s \n", addr, dataset,
		operation, indexType, process, partitions)

	client := createClient(addr)
	defer client.Close()
	if dataset == "taip" || dataset == "zc" {
		Dim = 768
	}else if dataset == "sift" {
		Dim = 128
	}
	if operation == "Insert" {
		Insert(client, dataset, indexType)
	}
	if operation == "Search" {
		Search(client, dataset, indexType, process, partitions)
	}
	if operation == "Index" {
		CreateIndex(client, dataset, indexType)
	}
	if operation == "Load" {
		Load(client, dataset, partitions)
	}
	if operation == "Release" {
		Release(client, dataset, partitions)
	}
	return
}
