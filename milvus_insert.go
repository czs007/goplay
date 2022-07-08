package main

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/sbinet/npyio"
	milvusClient "github.com/zhagnlu/milvus-sdk-go/v2/client"
	"github.com/zhagnlu/milvus-sdk-go/v2/entity"
)

var (
	NB               = 1000000
	ID               = 0
	PartitionNum     = 1
	PerFileRows      = 100000
	CurPartitionName = DefaultPartitionName
	PartitionCnt     = 0
)

func ReadFloatFromNumpyFile(dataPath string) [][]float32 {
	fmt.Println(dataPath)
	f, err := os.Open(dataPath)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	r, err := npyio.NewReader(f)
	if err != nil {
		panic(err)
	}

	fmt.Printf("npy-header:%v \n", r.Header)
	shape := r.Header.Descr.Shape
	raw := make([]float64, 0)

	err = r.Read(&raw)
	if err != nil {
		panic(err)
	}

	fmt.Println(len(raw))
	res := make([][]float32, 0)
	for j := 0; j < shape[0]; j++ {
		vec := make([]float32, 0)
		for i := 0; i < shape[1]; i++ {
			vec = append(vec, float32(raw[j*shape[1]+i]))
		}
		// fmt.Println(len(vec)
		res = append(res, vec)
	}
	return res
}

func InsertPipeline(client milvusClient.Client, dataset, indexType string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	pk := &entity.Field{
		Name:       "id",
		PrimaryKey: true,
		AutoID:     false,
		DataType:   entity.FieldTypeInt64,
	}
	vec := &entity.Field{
		Name:       VecFieldName,
		DataType:   entity.FieldTypeFloatVector,
		TypeParams: map[string]string{"dim": strconv.Itoa(Dim)},
	}
	schema := &entity.Schema{
		CollectionName: dataset,
		Description:    dataset,
		AutoID:         false,
		Fields:         []*entity.Field{pk, vec},
	}

	has, err := client.HasCollection(context.Background(), dataset)
	if err != nil {
		panic(err)
	}
	if has {
		fmt.Println("Collection exist, drop it.")
		if err = client.DropCollection(context.Background(), dataset); err != nil {
			panic(err)
		}
	}
	if err = client.CreateCollection(context.Background(), schema, 1); err != nil {
		panic(err)
	}

	fmt.Printf("create collection:%s done \n", dataset)
	partitionNames := make([]string, PartitionNum)
	for i := range partitionNames {
		partitionNames[i] = fmt.Sprintf("p%d", i)
	}
	partitionNames[0] = DefaultPartitionName
	for _, partition := range partitionNames {
		if partition == DefaultPartitionName {
			continue
		}
		if err = client.CreatePartition(context.Background(), dataset, partition); err != nil {
			panic(err)
		}
	}

	fmt.Printf("create partition:%+q done \n", partitionNames)
	go printInsertProgress(ctx)
	if dataset == "taip" || dataset == "zc" {
		// 1. Create index
		if indexType == "HNSW" {
			if err := client.CreateIndex(ctx, dataset, VecFieldName, NewTaipHNSWIndex(), false); err != nil {
				panic(err)
			}
		} else if indexType == "IVF_FLAT" {
			if err = client.CreateIndex(ctx, dataset, VecFieldName, NewTaipIVFFLATIndex(), false); err != nil {
				panic(err)
			}
		} else if indexType == "FLAT" || indexType == "" {
			// nothing to do
		}
		// 2. Insert data
		for i := 0; i < PartitionNum; i++ {
			for _, partition := range partitionNames {
				CurPartitionName = partition
				for j := 0; j < NB; j += PerFileRows {
					_, err = client.Insert(ctx, dataset, partition,
						generateInertData(TaipDataPath, PerFileRows, "Int64"), generateInertData(TaipDataPath, i, "FloatVector"))
					if err != nil {
						panic(err)
					}
					ID += PerFileRows
					PartitionCnt += PerFileRows
				}
			}
		}
	} else if dataset == "sift" {
		// 1. Create index
		if indexType == "HNSW" {
			if err = client.CreateIndex(ctx, dataset, VecFieldName, NewSiftHNSWIndex(), false); err != nil {
				panic(err)
			}
		} else if indexType == "IVF_FLAT" {
			if err = client.CreateIndex(ctx, dataset, VecFieldName, NewSiftIVFFLATIndex(), false); err != nil {
				panic(err)
			}
		} else if indexType == "FLAT" || indexType == "" {
			// nothing to do
		}
		fmt.Printf("create index:%s for collection:%s done \n", indexType, dataset)
		// 2. Insert data
		for i := 0; i < PartitionNum; i++ {
			for _, partition := range partitionNames {
				CurPartitionName = partition
				for j := 0; j < NB; j += PerFileRows {
					_, err = client.Insert(ctx, dataset, partition,
						generateInertData(SiftDataPath, PerFileRows, "Int64"), generateInertData(SiftDataPath, i, "FloatVector"))
					if err != nil {
						panic(err)
					}
					ID += PerFileRows
					PartitionCnt += PerFileRows
				}
			}
		}
	}
	fmt.Printf("insert data:(%d rows with index:%s) done \n", NB, indexType)
	return
}

func printInsertProgress(ctx context.Context) {
	timeTick := time.NewTicker(time.Second)
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\nInsert done!")
			return
		case <-timeTick.C:
			fmt.Printf("Partition:%-9s Inserting:[%8d/%9d] Total:[%-9d/%-9d]\r",
				CurPartitionName, PartitionCnt, NB, ID, NB*PartitionNum)
		}
	}
}

func generateInertData(dataPath string, step int, t string) entity.Column {
	var colData entity.Column
	switch t {
	case "Int64":
		intData := make([]int64, step)
		for i := ID; i < PerFileRows; i++ {
			intData[i] = int64(i)
		}
		colData = entity.NewColumnInt64("id", intData)
	case "FloatVector":
		colData = entity.NewColumnFloatVector(VecFieldName, Dim, generatedInsertEntities(dataPath, step))
	default:
		panic(fmt.Sprintf("column type %s is not supported", t))
	}
	return colData
}

func generatedInsertEntities(dataPath string, num int) [][]float32 {
	filePath := generateInsertPath(dataPath, num)
	return ReadFloatFromNumpyFile(filePath)
}

func Insert(client milvusClient.Client, dataset string, partitionNum int) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	partitionNames := make([]string, partitionNum)
	for i := range partitionNames {
		partitionNames[i] = fmt.Sprintf("p%d", i)
	}
	partitionNames[0] = DefaultPartitionName

	go printInsertProgress(ctx)
	if dataset == "taip" || dataset == "zc" {
		for i := 0; i < PartitionNum; i++ {
			for _, partition := range partitionNames {
				CurPartitionName = partition
				for j := 0; j < NB; j += PerFileRows {
					_, err := client.Insert(ctx, dataset, partition,
						generateInertData(TaipDataPath, PerFileRows, "Int64"), generateInertData(TaipDataPath, i, "FloatVector"))
					if err != nil {
						panic(err)
					}
					ID += PerFileRows
					PartitionCnt += PerFileRows
				}
			}
		}
	} else if dataset == "sift" {
		// 2. Insert data
		for i := 0; i < PartitionNum; i++ {
			for _, partition := range partitionNames {
				CurPartitionName = partition
				for j := 0; j < NB; j += PerFileRows {
					_, err := client.Insert(ctx, dataset, partition,
						generateInertData(SiftDataPath, PerFileRows, "Int64"), generateInertData(SiftDataPath, i, "FloatVector"))
					if err != nil {
						panic(err)
					}
					ID += PerFileRows
					PartitionCnt += PerFileRows
				}
			}
		}
	}

	fmt.Printf("insert data:(%d rows with index:%s) done \n", NB, indexType)
	return
}
