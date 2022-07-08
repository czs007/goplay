package main

import (
	"context"
	"fmt"
	"time"

	milvusClient "github.com/zhagnlu/milvus-sdk-go/v2/client"
)

var (
	CurLoadRows   = 0
	TotalLoadRows = 0
)

func Flush(client milvusClient.Client, dataset string) {
	ctx, _ := context.WithCancel(context.Background())
	client.Flush(ctx, dataset, false)
	fmt.Println("Flush Done")
}

func Load(client milvusClient.Client, dataset string, partitions []string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go printLoadProgress(ctx)
	if len(partitions) == 0 {
		err := client.LoadCollection(ctx, dataset, false)
		if err != nil {
			print(err)
		}
	} else {
		err := client.LoadPartitions(ctx, dataset, partitions, false)
		if err != nil {
			panic(err)
		}
	}

	confirmLoadComplete(client, dataset)
	return
}

func Release(client milvusClient.Client, dataset string, partitions []string) {
	if len(partitions) == 0 {
		if err := client.ReleaseCollection(context.Background(), dataset); err != nil {
			panic(err)
		}
	} else {
		if err := client.ReleasePartitions(context.Background(), dataset, partitions); err != nil {
			panic(err)
		}
	}
	return
}

func confirmLoadComplete(client milvusClient.Client, dataset string) {
	// TODO: sdk does not implement this function
}

func printLoadProgress(ctx context.Context) {
	ticker := time.NewTicker(time.Second)
	for {
		select {
		case <-ctx.Done():
			fmt.Println("\n Load done!")
			return
		case <-ticker.C:
			fmt.Print("loading...\r")
		}
	}
}

func GetCollectionInfo(client milvusClient.Client, dataset string) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	infos, err := client.GetQuerySegmentInfo(ctx, dataset)
	if err != nil {
		panic(err)
	}
	var allrows int64
	fmt.Printf("show collection:%s segment info: \n", dataset)
	for _, info := range infos {
		fmt.Printf("segment id:%d, partition id:%d, index name:%s, rows:%d, state:%s\n", info.ID, info.ParititionID, info.IndexName, info.NumRows, info.State.String())
		allrows += info.NumRows
	}
	fmt.Printf("collection load rows:%d \n", allrows)
}
