package main

import (
	"bufio"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"strconv"
	"strings"
	"time"

	milvusClient "github.com/zhagnlu/milvus-sdk-go/v2/client"
	"github.com/zhagnlu/milvus-sdk-go/v2/entity"
	"google.golang.org/grpc"
)

const (
	CollectionName       = "taip"
	DefaultPartitionName = "_default"
	RunCount             = 10000
	VecFieldName         = "vec"

	TaipDataPath = "/data/milvus/raw_data/zjlab"
	SiftDataPath = "/data/milvus/raw_data/sift"
	QueryFile    = "query.npy"
)

var (
	Dim = 768
)

func createClient(addr string) milvusClient.Client {
	opts := []grpc.DialOption{grpc.WithInsecure(),
		grpc.WithBlock(),                   //block connect until healthy or timeout
		grpc.WithTimeout(20 * time.Second)} // set connect timeout to 2 Second
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
	fileVecs := ReadFloatFromNumpyFile(filePath)
	vectors := make([]entity.Vector, 0)
	for i := 0; i < nq; i++ {
		var vector entity.FloatVector = fileVecs[i]
		//fmt.Println(len(vector))
		//fmt.Println(fileVecs[i])
		vectors = append(vectors, vector)
	}
	return vectors
}

func generateInsertFile(x int) string {
	return "binary_" + strconv.Itoa(Dim) + "d_" + fmt.Sprintf("%05d", x) + ".npy"
}

func generateInsertPath(dataPath string, x int) string {
	return path.Join(dataPath, generateInsertFile(x))
}

func setDimByDataSet(dataSet string) {
	switch dataset {
	case "taip", "zc":
		Dim = 768
	case "sift":
		Dim = 128
	default:
		panic("not known dataset:" + dataSet)
	}
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
	addr         string
	dataset      string
	partitionNum int
	indexType    string
	process      int
	operation    string
	partitions   []string
)

func init() {
	flag.StringVar(&addr, "host", "127.0.0.1:19530", "milvus addr")
	flag.StringVar(&dataset, "dataset", "taip", "dataset for test")
	flag.StringVar(&indexType, "indexType", "FLAT", "index type for collection, HNSW | IVF_FLAT | FLAT")
	flag.StringVar(&operation, "op", "", "what do you want to do")
	flag.Var(newSliceValue([]string{}, &partitions), "p", "partitions which you want to load")
	flag.IntVar(&partitionNum, "partitionNum", 1, "collection's partition num")
	flag.IntVar(&process, "process", 1, "goroutines for test")
}

func main() {
	flag.Parse()
	fmt.Printf("host: %s, dataset: %s, operation: %s, index type: %s, process: %d, partitions: %s \n", addr, dataset,
		operation, indexType, process, partitions)

	client := createClient(addr)
	defer client.Close()

	setDimByDataSet(dataset)

	switch operation {
	case "CreateCollection":
		CreateCollection(client, dataset, partitionNum)
	case "Insert":
		Insert(client, dataset, partitionNum)
	case "Search":
		Search(client, dataset, indexType, process, partitions)
	case "CreateIndex":
		CreateIndex(client, dataset, indexType)
	case "Load":
		Load(client, dataset, partitions)
	case "Release":
		Release(client, dataset, partitions)
	case "InsertPipeline":
		InsertPipeline(client, dataset, indexType)
	case "CollectionStatus":
		GetCollectionInfo(client, dataset)
	case "Flush":
		Flush(client, dataset)
	default:
		panic("not supported operation:" + operation)
	}

	return
}
