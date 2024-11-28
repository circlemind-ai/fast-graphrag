# EDGE TAGGING FOR GRAPH PRUNING, WEIGHTED EXPLORATION, AND GLOBAL QUERIES

## Project overview

We want to add tags to relationships, similar to how LightRAG does, in order to introduce several new capabilities:
- graph pruning (eventually) -> given a query remove edges whose tags are completely irrelevant
- weighted exploration -> personalised page rank allow to specify weights for the edges. We believe this to be helpful to allow the exploration to focus towards nodes more relevant to the query. More on this later.
- global queries -> to answer global queries (such as `what are the main themes in the story?`), LightRAG selectes the top k edges based on some abstract attributes (or TAGS) extracted from the query such as `STORY THEME` or `PLOT DEVELOPMENTS`, we can try to do the same but, since we run pagerank, we can actually select all the edges that match those attributes and then run pagerank to converge to the more relevant nodes.


### Weighted exploration in detail
Consider the example `who is the girlfriend of Harry Potter?`. `Harry Potter` is a very central entity in the knowledge graph, consequently it will have a very high rank (number of edges connected to the node). Given how PageRank works all, the nodes connected to `Harry Potter` will have the same weight, so that Ginny will be indistinguishable from Hermiony or Ron (who could actually even have an higher score given that they are overall nodes with likely higher rank). So we need to weight connections!

However, it would be impractical (and extremely costly) to embed (at insertion time) and weight (at query time) all the edges in a graph. A possible solution is tagging: the idea is to associate x tags to each relation (extracted via LLM together with its description); at query time, we can weight each tag given the query and then compute the weight of each relation as the sum of the weights of its associated tags.

### Implementative thoughts

Some thoughts:
- we do not want too many tags (#edges  << #tags) otherwise we could just embed the edges. How can we achieve this? We need to somehow incrementally cluster new tags with existing ones.