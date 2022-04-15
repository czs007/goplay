package main

import (
	"context"
	"fmt"
	milvusClient "github.com/xiaocai2333/milvus-sdk-go/v2/client"
	"github.com/xiaocai2333/milvus-sdk-go/v2/entity"
	"time"
)

var (
	NB = 1000000
	ID = 0
	PartitionNum = 1
	PerFileRows = 100000
	CurPartitionName = DefaultPartitionName
	PartitionCnt = 0
)

func Insert(client milvusClient.Client, dataset, indexType string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go printInsertProgress(ctx)

	pk := &entity.Field{
		Name: "id",
		PrimaryKey: true,
		AutoID: false,
		DataType: entity.FieldTypeInt64,
	}
	vec := &entity.Field{
		Name: VecFieldName,
		DataType: entity.FieldTypeFloatVector,
		TypeParams: map[string]string{"dim": "768"},
	}
	schema := &entity.Schema{
		CollectionName: dataset,
		Description: "dataset",
		AutoID: false,
		Fields: []*entity.Field{pk, vec},
	}

	has, err := client.HasCollection(context.Background(), dataset)
	if err != nil {
		panic(err)
	}
	if has {
		if err = client.DropCollection(context.Background(), dataset); err != nil {
			panic(err)
		}
	}
	if err = client.CreateCollection(context.Background(), schema, 2); err != nil {
		panic(err)
	}

	partitionNames := make([]string, PartitionNum)
	for i := range partitionNames {
		partitionNames[i] = fmt.Sprintf("p%d", i)
	}
	partitionNames[0] = DefaultPartitionName
	for _, partition := range partitionNames {
		if err = client.CreatePartition(context.Background(), dataset, partition); err != nil {
			panic(err)
		}
	}
	if dataset == "taip" {
		for i := 0; i < PartitionNum; i++ {
			for _, partition := range partitionNames {
				CurPartitionName = partition
				for j := 0; j < NB; j+=PerFileRows {
					_, err = client.Insert(context.Background(), CollectionName, DefaultPartitionName,
						generateInertData(PerFileRows, "Int64"), generateInertData(i, "FloatVector"))
					if err != nil {
						panic(err)
					}
					ID += PerFileRows
				}
			}
		}
	}
	if dataset == "sift" {
		//TODO:
	}
}

func printInsertProgress(ctx context.Context) {
	timeTick := time.NewTicker(time.Second)
	for {
		select {
		case <-ctx.Done():
			return
		case <-timeTick.C:
			fmt.Printf("Partition:%-9s Inserting:[%8d/%9d] Total:[%-9d/%-9d]\r",
				CurPartitionName, PartitionCnt, NB, ID, NB*PartitionNum)
		}
	}
}
