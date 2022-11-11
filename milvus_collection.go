package main

import (
	"context"
	"fmt"
	"strconv"

	milvusClient "github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"
)

func CreateCollection(client milvusClient.Client, dataset string, partitionNum int) {
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
	partitionNames := make([]string, partitionNum)
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
}
