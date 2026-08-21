# **大規模言語モデルにおけるグラフ構造を活用した検索拡張生成（Graph-Based RAG）の最前線：既存の分類体系の拡張と新興アーキテクチャの包括的分析**

## **1\. 序論**

大規模言語モデル（Large Language Model: LLM）の急速な発展に伴い、外部の知識源から動的に情報を取得して回答を生成する検索拡張生成（Retrieval-Augmented Generation: RAG）は、知識集約型タスクにおける事実上の標準アプローチとして広く定着している。しかしながら、従来のテキストチャンクに基づく密ベクトル類似度検索（Dense Retrieval）は、複数文書にまたがるエンティティ間の関係性を辿る多段推論（Multi-hop Reasoning）や、情報が断片化して散在しているタスクにおいて、文脈の断絶や関連情報の想起漏れを引き起こすという根本的な限界を抱えている。  
これらの課題を克服するため、非構造化テキストから自動的にエンティティや関係性を抽出し、ナレッジグラフ（Knowledge Graph: KG）を構築した上で探索および推論を行う「Graph-Based RAG（GraphRAG）」が急速に台頭している。提出された原稿（main.tex）では、代表的な5系統のGraph-Based RAG（GraphRAG、LightRAG、PathRAG、HippoRAG、YoutuRAG）を取り上げ、KGの中間品質（エンティティ抽出精度、関係抽出精度、グラフ密度など）が下流の検索性能や回答品質に及ぼす影響を定量的に分析している1。同原稿における最大の貢献の一つは、「グラフ密度の量単独では性能向上に直結せず、推論連鎖を構成するキー実体（ブリッジ実体）の網羅性が重要である」という因果的メカニズムを実証した点にある1。  
しかしながら、Graph-Based RAGの研究領域は現在、爆発的な進化の過程にあり、原稿で議論されている5つの手法は、全体像の一部を構成するに過ぎない。近年では、グラフ構築における関係抽出の計算コストと抽出不安定性を根本から見直す手法（LinearRAG、E2-GraphRAG）、グラフニューラルネットワーク（GNN）を統合して構造的推論を強化しモダリティギャップを解消する手法（GNN-RAG、S-Path-RAG）、認知科学的アプローチやエージェントベースの動的探索を取り入れた手法（StructRAG、Think-on-Graph、MemGraphRAG）、さらには医療や因果推論などの特定ドメインに特化した階層的アーキテクチャ（MedGraphRAG、CausalKG）など、多様なパラダイムが次々と提案されている2。  
本報告書は、原稿の「2 関連研究」における理論的枠組みを大幅に深掘りし、議論の対象となっていない10以上の最新のGraph-Based RAGパラダイムを網羅的に調査・体系化するものである。アルゴリズムの設計思想、インデックス構築のスケーラビリティ、ニューラル手法とのアーキテクチャレベルの融合、およびエージェントベースの推論メカニズムを比較分析し、Graph-Based RAGの進化の軌跡と次世代システム設計における高次な洞察を提示する。

## **2\. 原稿における「関連研究」の理論的基盤と再解釈**

原稿において評価対象とされた5つの手法は、グラフの構築方式と検索パラダイムの観点から、Graph-Based RAGの初期から中期にかけての進化を代表するアーキテクチャである1。これらのアーキテクチャは、それぞれ異なるアプローチで従来のRAGの限界に対処している。

* **階層的コミュニティ要約型（GraphRAG）**: Microsoftによって提唱されたこの手法は、Leidenアルゴリズムを用いてノードをコミュニティに分割し、各コミュニティの要約を事前生成する1。広範なグローバルクエリに対して俯瞰的な回答を生成する能力に長けているが、コミュニティ要約の生成に膨大なLLMトークンを消費するという重大なコスト面の課題がある9。  
* **二層インデックス型（LightRAG）**: エンティティレベルの局所的（Low-level）検索と、関係性・テープレベルの広域的（High-level）検索をデュアルで実行する。インデックスの増分更新が容易であり、検索の高速化と包括性を両立している1。  
* **関係パス探索型（PathRAG）**: クエリに関連するエンティティ間の関係パス（Relational Path）を抽出し、枝刈り（Pruning）を行うことで、推論に必要なコンテキストのみをLLMのプロンプトに統合する。これはグラフ内のノイズを削減し、LLMのコンテキスト長を節約するための有効なアプローチである1。  
* **神経生物学着想型（HippoRAG）**: 大脳新皮質と海馬の相互作用に着想を得ている。OpenIEによって抽出された実体に、ベクトル類似度に基づくKNN同義エッジ（Synonymy Edge）を追加し、Personalized PageRank（PPR）アルゴリズムを用いて推論パスを探索する。原稿の実験でも最も高い検索精度と頑健性を示した1。  
* **スキーマ駆動・進化型（YoutuRAG）**: 事前定義されたオントロジースキーマに基づいて抽出を行い、未知の概念型を発見した際にスキーマを動的に進化（Evolution）させる。ドメイン特化型の知識を構造化する上で強力な手法である1。

原稿で言及されているXiaoらによる「GraphRAG-Bench」の分類（Tree構造、Passage Graph、Knowledge Graph、Rich Knowledge Graph）は、計算コストと構造組織化の品質のトレードオフを明確にした1。しかし、現在の最新研究は、これらのトレードオフ自体を破壊する新しいアーキテクチャ（ゼロトークン構築や非関係型グラフ）へと移行している。以降のセクションでは、原稿の範囲を超えた新興手法群について詳細に論じる。

## **3\. エッジ抽出のパラダイムシフト：関係性の排除と計算量の線形化**

Graph-Based RAGにおける最大のボトルネックは、非構造化テキストからの知識抽出フェーズにおける「LLMトークン消費の膨大さ」と「関係抽出の不安定性（幻覚によるノイズの混入）」である6。原稿の分析においても、関係想起率（Relation Recall: RR）が不要なエッジの増加（ノイズ）を通じて回答品質に負の影響を与える可能性が示唆されている1。この課題を根本的に解決するため、グラフのトポロジー設計自体を抜本的に見直す手法が登場している。

### **3.1 LinearRAG：関係抽出を排除した線形スケールグラフ**

従来のGraphRAGは、エンティティ間の「関係（エッジ）」をLLMに推論させることで知識グラフを構築していた。しかし、関係抽出は幻覚（ハルシネーション）の温床となりやすく、抽出コストも極めて高い6。LinearRAGは、この関係抽出を完全に排除し、軽量な固有表現抽出（NER）と意味的リンキングのみを利用する「Tri-Graph（エンティティ、文、パッセージの3層構造）」を提案した6。  
LinearRAGにおける関係を持たないグラフは、以下の隣接行列によって数学的に定義される。パッセージ集合 ![][image1]、文集合 ![][image2]、エンティティ集合 ![][image3] に対し、Contain行列 ![][image4] とMention行列 ![][image5] は次のように定式化される：  
![][image6]  
![][image7]  
この手法は、関係のラベルを持たない代わりに、文脈内の共起関係をスパース行列として保持する。検索時は、(1) 局所的な意味的ブリッジングによる関連エンティティの活性化、(2) グローバルな重要度集約によるパッセージの抽出、という2段階戦略を取る6。 このアプローチの最大の洞察は、「LLMの強力な文脈理解能力を前提とすれば、明示的な関係ラベル（例: A is\_capital\_of B）をインデックス側で持たずとも、関連するエンティティが共起するパッセージ群を正確に検索できれば推論は成功する」という点である。これにより、インデックス構築時のLLMトークン消費をゼロにし、SpaCyなどの軽量な言語モデルを利用することで構築時間を77%以上削減しつつ、複雑な多段推論において従来のGraphRAGを凌駕する精度（例：2WikiMultiHopQAデータセットにおいて63.70%のGPTベース精度）を達成している6。

### **3.2 E2-GraphRAG（![][image8]GraphRAG）：効率性と有効性のストリームライン化**

GraphRAGが抱えるコミュニティ要約の非効率性に対処するため、E2-GraphRAGは、インデックス構築と検索の両面を最適化している16。 構築フェーズでは、文書チャンクからLLMを用いて「要約ツリー（Summary Tree）」を構築するプロセスと、SpaCy等を用いて「エンティティグラフ（Entity Graph）」を構築するプロセスを分離し、両者の間に双方向インデックスを構築する。これにより、エンティティとチャンクの多対多の関係を高速にルックアップできる17。 検索フェーズでは、クエリに応じてローカル検索（エンティティ中心）とグローバル検索（コミュニティ要約中心）を動的に切り替える適応型検索戦略（Adaptive Retrieval）を導入している。結果として、GraphRAGと比較してインデックス構築で10倍、LightRAGと比較して検索実行で100倍の高速化を達成しつつ、同等以上のQA性能を維持している16。

### **3.3 NodeRAG：異種ノードによる粒度制御とグラフ構造の最適化**

情報が粗視化されすぎる（ノイズが多くなる）という既存の同種グラフ（Homogeneous Graph）の欠点に対処するため、NodeRAGは異種グラフ（Heterogeneous Graph）アーキテクチャを採用している18。従来のGraph-Based RAG手法がグラフ構造の設計自体を軽視してきたのに対し、NodeRAGはグラフアーキテクチャをRAG性能を決定づける基盤的要素と位置づけている19。  
NodeRAGは、グラフ分解・拡張・強化の3段階を通じてヘテログラフを構築する：

> 1. **グラフ分解（Graph Decomposition）**: LLMがテキストチャンクを「意味単位（Semantic Units: ![][image9]）」「エンティティ（Entities: ![][image10]）」「関係性（Relationships: ![][image11]）」に分解し、初期ヘテログラフ ![][image12] を形成する18。  
> 2. **ノード重要度に基づく拡張（Node Importance-Based Augmentation）**: K-core分解（![][image13]）と媒介中心性（Betweenness Centrality: ![][image14]）を用いて構造的に重要なキーエンティティを特定する。LLMはこれらの重要エンティティに対する属性要約（Attribute summaries: ![][image15]）を生成し、グラフ ![][image16] として追加する18。  
> 3. **コミュニティ検出に基づく集約（Community Detection-Based Aggregation）**: ![][image16] に対してLeidenアルゴリズムを適用し、コミュニティ内の高次元な要素（High-level elements: ![][image17]）や全体的な概要（Overviews: ![][image18]）を抽出してノード化する。これらはK-meansクラスタリングによって意味的にリンクされる18。

この機能的に区別された7種類の異種構造により、NodeRAGは「局所的な検索」と「大局的な要約検索」のワークフローの不一致を解消し、クエリの性質に応じて適切な粒度のノードを正確かつ階層的にナビゲートすることが可能となっている19。

## **4\. ニューラル検索とグラフアルゴリズムの深層融合（モダリティギャップの解消）**

原稿において、HippoRAGがPPR（Personalized PageRank）を用いたように、グラフのトポロジーを利用した検索アルゴリズムは極めて重要である。近年では、これをさらに推し進め、グラフニューラルネットワーク（GNN）を用いてグラフ上の潜在的な推論パスを学習・抽出するハイブリッドなアプローチが主流となりつつある。これらは、テキストベースの埋め込みモデルだけでは捉えきれない構造的文脈をニューラルネットワークに学習させる試みである。

### **4.1 GNN-RAG：Dense Subgraph推論と最短経路の言語化**

GNN-RAGは、大規模言語モデルの自然言語理解能力と、GNNの複雑な構造推論能力を融合させた先駆的なフレームワークである23。数十万から数百万の事実を格納する大規模ナレッジグラフにおいて、無関係な情報をLLMに与えると推論を大きく阻害する（ハルシネーションの誘発）23。  
GNN-RAGは以下のステップで動作する：

> 1. クエリのエンティティをシードとして、PageRank等により密なサブグラフ（Dense Subgraph）を抽出する25。  
> 2. 抽出されたサブグラフに対し、多層GNN（例：ReaRevアーキテクチャ）を適用する。各ノードは隣接情報とクエリの埋め込みメッセージを伝播し、最終的にクエリの回答となる確率分布をノードごとに出力する25。  
> 3. 高い確率を持つ候補ノードに対し、シードエンティティからの「最短経路（Shortest Paths）」を抽出する24。  
> 4. 抽出されたパスを自然言語のテキストに変換（Verbalization）し、LLMへのプロンプトコンテキストとして入力する23。

さらに、GNN-RAGはLLMベースの検索器とGNNを組み合わせる検索拡張（Retrieval Augmentation: RA）技術を導入している。実験において、GNN-RAGはWebQSPおよびCWQデータセットにおいて、テキストベースのRAGシステムと比較してF1スコアを8.9〜15.5ポイント向上させ、7Bクラスの小規模LLMでありながらGPT-4に匹敵、あるいはそれを凌駕する多段推論能力を実現した23。

### **4.2 S-Path-RAG：ソフト潜在パスの直接注入（Soft Latent Path Injection）**

GNN-RAGにおける「抽出されたパスを自然言語化する」というアプローチは、LLMが解釈しやすい反面、パスが長くなるとトークン消費が爆発的に増加し、LLMのコンテキストウィンドウを圧迫するという欠点がある29。また、トポロジー情報を一次元のテキストシーケンスに無理やり押し込むため、情報損失が生じる。  
S-Path-RAGは、意味的・構造的に重み付けされたk-最短パス、ビームサーチ、および制約付きランダムウォークを組み合わせたハイブリッド戦略で候補パスを列挙する29。そして最大の技術的飛躍は、抽出された長いパスリストをテキストとしてプロンプトに挿入するのではなく、パスの潜在表現（Latent representations）を生成し、クロスアテンション（Cross-attention）を通じてLLMの内部状態に直接注入（Soft Latent Path Injection）する点である29。 さらに、S-Path-RAGは「Neural-Socratic Graph Dialogue」と呼ばれる反復的なループを採用しており、LLMが不確実性を示した場合に、LLMが生成した診断メッセージをグラフのシード拡張やターゲット編集にマッピングして適応的な再検索を行う29。このアプローチにより、LLMは構造的証拠をトークンを消費することなく参照でき、グラフモダリティとテキストモダリティ間の壁（Modality Gap）を完全に橋渡しすることに成功している。

### **4.3 NGM-RAG：ニューラルグラフマッチングの導入**

NGM-RAG（Neural Graph Matching based RAG）は、検索対象となるグラフのノードをいかに正確に特定するかに焦点を当てている31。従来のGraphRAGでは、テキストの類似度マッチングに依存していたが、同義語や文脈に依存した抽象的な関連性を見落としやすい。 NGM-RAGは、以下の3つのスコアを適応的に重み付けして統合する31：

> 1. **直接マッチング（Direct Matching）**: Levenshtein距離を用いたノード名の字句レベルの一致。  
> 2. **テキスト類似度（Text Similarity）**: ノードの記述に対するTF-IDFやBM25による意味的類似度。  
> 3. **ニューラルグラフマッチング（Neural Graph Matching）**: LightGCNなどのパラメータフリーなGNNを用いた、構造的コンテキストに基づくノードの捕捉。

これにより、単なる語彙的重複を超えた、ノードの近傍構造やグラフ全体のトポロジーを考慮した精密なノード特定が可能となっている31。

| モデル名 | 検索アーキテクチャの核心 | グラフとLLMの統合方式 | 主な特徴と利点 |
| :---- | :---- | :---- | :---- |
| **GNN-RAG** | ReaRev GNN \+ 最短経路抽出 | 抽出パスの自然言語化 (Verbalization) | 多段推論における大幅なF1向上 (8.9\~15.5pt)23 |
| **S-Path-RAG** | K-最短パス \+ ビームサーチ | クロスアテンションによる潜在表現注入 | トークン消費の削減、Modality Gapの解消29 |
| **NGM-RAG** | LightGCNを用いたグラフマッチング | 類似度・BM25・GNNのアンサンブル | 同義語や文脈に依存した精緻なノード特定31 |

## **5\. エージェントベースの動的探索と認知科学的アプローチ**

検索フェーズを事前に定義された一度の処理（シングルパス）で終わらせるのではなく、LLM自身を自律的なエージェントとして機能させ、グラフ上を反復的に探索させるアプローチも近年の大きなトレンドである。これは、Graph-Based RAGを単なる情報抽出パイプラインから、自律的推論システムへと昇華させる試みである。

### **5.1 Think-on-Graph (ToG)：LLM ![][image19] KG パラダイムとビームサーチ**

Think-on-Graph（ToG）は、静的な検索をLLMに渡す「LLM ![][image20] KG」ではなく、LLMとナレッジグラフが密結合して探索を行う「LLM ![][image19] KG」パラダイムを提唱している32。 ToGのプロセスは以下の通りである：

> 1. **探索の初期化**: LLMがクエリからトピックエンティティを特定し、グラフ探索の起点（初期ビーム）とする32。  
> 2. **探索と推論の反復（ビームサーチ）**: LLMをエージェントとして呼び出し、グラフの隣接関係を評価させる。LLMは次に進むべき最も有望なトリプルをランキングし、関連性の低いパスを枝刈り（Prune）しながら推論パスを構築していく4。  
> 3. **回答生成**: 収集された情報がクエリに答えるのに十分とLLMが判断した場合、最終回答を生成する32。

ToGの利点は、推論経路が完全に追跡可能（Traceability）であり、かつ事前学習などの追加コストを必要としない点である4。特に、ToGをLLaMA-2-70Bのようなオープンモデルに適用した場合、標準的なChain-of-Thought（CoT）を用いたGPT-4の精度（CWQデータセットで46.0%）を上回る精度（57.6%）を達成し、劇的なコスト削減の可能性を示した32。  
しかし、その後の研究によりToGの限界も明らかになっている。ToGのような局所的な評価に基づくビームサーチは、「早期の近視眼的なコミットメント（Trap selection）」に陥りやすく、探索が真の論理的連鎖から逸脱する「セマンティック・ドリフト（Semantic Drift）」を引き起こすことが指摘されている35。さらに、ToGは巨大なLLMでは機能するが、小規模な言語モデル（SLM）に適用すると、ベースラインのCoTよりも性能が低下する場合があるという制約が確認されている34。これに対する解決策として、将来の軌道を予測しながら探索を行うFLARE等の強化学習的フレームワークの統合が進められている36。

### **5.2 StructRAG：認知科学に基づくハイブリッド情報の構造化**

情報の断片化が著しい知識集約型タスクにおいては、外部知識を「グラフ」という単一の表現形式に固定すること自体が制約となる場合がある。StructRAGは、「人間は知識集約的な推論を行う際、生の情報を表、ツリー、グラフなどの多様な構造的知識へと変換して処理する」という認知理論に着想を得たフレームワークである7。  
StructRAGは以下の3つのコンポーネントで構成される：

> 1. **ハイブリッド構造ルーター (Hybrid Structure Router)**: クエリの性質に基づいて、タスクに最適な構造タイプを動的に特定する。このルーターは、Direct Preference Optimization (DPO) ファインチューニングを通じて、人間の意思決定を模倣するように訓練されている37。  
> 2. **分散知識構造化器 (Scattered Knowledge Structurizer)**: 検索された生の断片的な文書群を、ルーターが選択した最適な構造フォーマット（表、グラフなど）へと再構築する37。  
> 3. **構造化知識利用器 (Structured Knowledge Utilizer)**: 複雑な質問を分解し、再構築された構造を利用して多段推論を実行する37。

関連する研究として、CoRE（Contrastive Retrieval-Augmented Generation on Experience）フレームワークなどが挙げられる。これらはモンテカルロ木探索（MCTS）を用いてエージェントの推論の「成功」と「失敗」の経験メモリ（Experience Memory）を構築し、対照学習的なインコンテキスト学習（ICL）を行うことで、構造的推論能力を飛躍的に向上させている39。StructRAGは、情報表現をタスクに適応させる点で、非常に柔軟かつ強力なアーキテクチャである38。

### **5.3 MemGraphRAGとマルチエージェント協調システム**

GraphRAGにおけるグラフ構築フェーズのもう一つの課題は、長大なコーパスをチャンク分割して独立に処理するため、コーパス全体にわたる文脈の整合性や論理的な一貫性が失われ、グラフが断片化しやすいことである5。  
MemGraphRAG（Memory-based Multi-Agent System for Graph RAG）は、この問題に対処するため、グラフ抽出プロセス全体を通じてグローバルな文脈を維持する「共有メモリ（Shared Memory）」によってサポートされたマルチエージェント社会（Collaborative society of agents）を導入している5。エージェントたちは個別のチャンクから情報を抽出する際に共有メモリを参照し、エンティティの重複、テーマの不一致、論理的な矛盾を動的に解決（Adjudication / 裁定）する。これにより、構造的な接続性が維持された、論理的に一貫性のある高品質なグローバルナレッジグラフの構築が可能となっている5。

## **6\. 特定ドメインにおける応用と知識表現の特化**

Graph-Based RAGの実用化において最も進展が見られるのが、高度な専門知識、情報の出処の追跡可能性（Traceability）、および論理的厳密性が求められる医療分野や因果推論ドメインである。原稿（main.tex）においても因果推論データセット（Causal-Reasoning-QA）が用いられているが、因果推論をグラフとして明示的にモデル化する試みは急速に拡大している。

### **6.1 Causal-KGと因果関係の明示的推論**

LLMは統計的な相関関係と因果関係（Causation）を混同する「因果的誤謬（Causal fallacies）」に陥りやすいことが指摘されている2。この問題に対し、因果関係を明示的な有向エッジとして表現する「Causal Knowledge Graph (CausalKG)」の構築が進められている。 たとえば、Fujitsu Causal Knowledge Graphは、論理的推論に基づくデータ駆動型の意思決定を支援するために、介入的推論（Interventional reasoning）や反事実的推論（Counterfactual reasoning）をグラフ上で実行可能にしている2。また、Lyuら（2023）は、糖尿病性腎症の臨床意思決定支援システムにおいて、SemMedDBやUpToDateなどの医学文献から抽出した知識を用いてCausalKGを構築し、共起率（Co-occurrence ratio）や因果率（Causality ratio）といった枝刈り戦略を適用することで、グラフのノイズを低減し診断予測の精度（AUCおよびF1スコア）を大幅に向上させた2。因果推論ドメインにおけるGraphRAGは、単純な事実の検索を超えて、「もしAが行われなかったらBはどうなるか」という高度な推論（What-if分析）を可能にする基盤として機能している。

### **6.2 MedGraphRAG：階層的医療知識の統合とU-Retrieval**

MedGraphRAGは、医療領域におけるLLMの安全性と信頼性を高めるために設計された特化型フレームワークである3。医療データ特有の複雑さを処理するため、文書をチャンク化する際に静的文字分離とトピックベースの意味的セグメンテーション（命題転送やスライディングウィンドウ技術）をハイブリッドで適用し、その後「3層の階層的グラフ構造（Triple Graph Construction）」を構築する3。

> 1. **ボトムレベル**: UMLS（Unified Medical Language System）などの医療辞書から得られる明確な医療用語と定義43。  
> 2. **ミドルレベル**: 医学の教科書や査読付き論文から得られる基盤的な医学的知識43。  
> 3. **トップレベル**: 電子カルテや医療レポートなどのプライベートなユーザー文書43。

検索においては、「U-Retrieval」と呼ばれる独自の戦略を採用している。これは、クエリから抽出されたキーワードを用いてグラフの階層タグをトップダウンでトラバースし、その後、抽出された関連エンティティと関係性からボトムアップで情報を合成して回答を生成するハイブリッドなメカニズムである3。MedGraphRAGは、生成された回答に根拠となる情報源（Source citations / Provenance）を明確に結びつけることができ、医療現場における解釈可能性（Interpretability）の要求に合致している3。

### **6.3 Co-MedGraphRAG：未知変数の明示的モデリング**

医療診断や意思決定においては、「何が分かっているか」と同等以上に「情報が欠落していること」自体が重要な推論のトリガーとなる。Co-MedGraphRAGは、「不完全な知識グラフ」を動的に構築するというユニークなアプローチを取る46。 ユーザーの医療クエリを受け取ると、システムはまず、既知の事実と「未知の変数（Pending entities）」を明示的に区別する。この未知の変数は特殊なプレースホルダー「![][image21]（none）」としてグラフ内にモデル化される46。その後、外部のグローバルナレッジグラフとのハイブリッド推論を通じて、この欠落したプレースホルダーを補完（Knowledge graph completion）していく。このパラダイムは、受動的な情報検索から、「質問に特化した積極的な知識構築」へとRAGを昇華させるものである46。  
さらに、コード生成やロボティクスといった他のドメインでも、グラフ表現の特化が進んでいる。たとえば、コード補完においては制御フローやデータ依存関係を捉えるコードコンテキストグラフ（GraphCoder）が、ロボットタスク計画においては3Dシーングラフ（SayPlan）や状態遷移グラフ（AVIS）が活用されており、Graph-Based RAGの適用範囲は自然言語処理の枠を大きく越えて拡大している47。

## **7\. 比較分析と次世代システム設計への高次な洞察（Second and Third-Order Insights）**

原稿（main.tex）での実証的分析結果と、本報告書で調査した最新のGraph-Based RAG技術群のアーキテクチャ的特徴を総合することで、今後のシステム設計における極めて重要な高次な洞察（メカニズムと波及効果）が浮かび上がる。以下の表は、各手法のアプローチの方向性を要約したものである。

| 開発の方向性 | 代表的モデル | 解決しようとした根本的課題 | アプローチの核心 | 予想される波及効果 |
| :---- | :---- | :---- | :---- | :---- |
| **計算量の線形化と関係の排除** | LinearRAG, E2-GraphRAG | グラフ構築時のLLMトークン爆発、関係抽出におけるハルシネーション | 明示的関係ラベルの排除、行列ベースの共起抽出、SpaCy等の軽量NER | LLM APIコストの劇的削減と、インデックスのリアルタイム増分更新の実現 |
| **ニューラル空間でのアーキテクチャ融合** | GNN-RAG, S-Path-RAG | テキストモダリティへの変換（Verbalization）に伴う情報損失、コンテキスト長超過 | GNNによるパス抽出、潜在ベクトル（Latent）のLLM推論レイヤへの直接注入 | 大規模グラフにおける多段推論の超高速化と、トークン消費を伴わない知識参照 |
| **自律的エージェントと認知ルーティング** | ToG, StructRAG, MemGraphRAG | 静的インデックスの硬直性、チャンク分割に伴うグローバルな論理矛盾 | ビームサーチ、タスクに応じた構造の動的選択、共有メモリを通じたマルチエージェント裁定 | 静的検索から、知識の「自己学習・自己訂正機能」を持った動的推論プラットフォームへの進化 |

**洞察1：知識抽出コストと検索精度のトレードオフの崩壊（「関係」から「共起構造」へのパラダイムシフト）** 原稿において、関係想起率（Relation Recall: RR）が不要なエッジの増加（ノイズ）を通じて回答F1と負の関連を示す可能性が示唆された。この問題に対する究極の解決策がLinearRAGの設計思想に見られる1。LinearRAGは関係抽出の試みを完全に放棄し、文脈内での「エンティティと文・パッセージの共起行列（Tri-Graph）」へと知識表現をダウングレードさせた6。一見するとこれは知識の退化に見えるが、実際にはLLMのコンテキスト理解能力が十分に高いため、関係ラベル（Edge Type）がなくても、関連するエンティティが共起するパッセージ群を与えれば、LLM自身がその場で関係性を推論できることを証明している。これにより、「グラフ抽出には膨大なLLMのコストがかかる」というGraphRAG最大のボトルネックが破壊され、ゼロトークンでの線形スケーラブルなインデックス構築が可能となっている。これは、Graph-Based RAGのシステム設計において「構築コストをかけて精緻な知識グラフを作る」方向から「軽量なインデックスを用いてLLMの推論能力に委ねる」方向への揺り戻し（ハイブリッド化）が起きていることを示している。  
**洞察2：シンボリック検索からマルチモーダル・ニューラル表現への移行（Modality Gapの解消）** 原稿のHippoRAGや、初期のGNN-RAGは、グラフというシンボリックな構造からパスを見つけ出し、それを自然言語のテキスト（プロンプト）に変換してLLMに入力する。しかし、この「言語化（Verbalization）」プロセスは、複雑なトポロジー情報を一次元のテキストシーケンスに無理やり押し込むため、情報損失やコンテキストウィンドウの枯渇を招く25。S-Path-RAGが示す「パスの潜在表現（Soft Latent）のクロスアテンション注入」は、ナレッジグラフとLLMのモダリティの壁（Modality Gap）を取り払うものである29。これは、Graph-Based RAGが単なる「外部ツールの呼び出し（プロンプトエンジニアリングの延長）」から、GNNとLLMの表現空間を直接結合する「統合的アーキテクチャ」へと進化していることを示唆している。  
**洞察3：近視眼的なグラフ探索の限界とエージェント間コンセンサスの必要性** Think-on-Graph (ToG)のような動的ビームサーチは、グラフ上を自己回帰的に探索する有望な手法であるが、ローカルなヒューリスティクスに依存するため、一度誤ったパスに進むと軌道修正が困難になる（セマンティック・ドリフト）という限界が明らかになった35。また、グラフ構築フェーズにおいても、抽出をチャンク単位に依存すると、同じ概念が異なる名前で登録されるなどのグローバルな不整合が生じる。 これに対する解決策は、MemGraphRAGが示すような「共有メモリを持つマルチエージェント協調システム」である5。抽出時も検索時も、単一のプロセスによる決定ではなく、複数のエージェントが過去の経験（メモリ）を参照しながら討議（Adjudication）を行うアプローチである。これは、RAGシステムが単なる「抽出パイプライン」から、より人間に近い「集合知的な認識・推論プロセス」へとパラダイムシフトしていることを明確に示している。

## **8\. 結論**

本報告書では、原稿（main.tex）において言及された5つの初期・中期Graph-Based RAGモデル（GraphRAG, LightRAG, PathRAG, HippoRAG, YoutuRAG）を出発点として、その限界を克服するために登場した10以上の最新アーキテクチャ群を網羅的に分析した。  
原稿の実証分析が示唆した「グラフ密度の量単独では性能向上に直結せず、ノイズが推論の妨げとなる」という洞察は、LinearRAGによる関係抽出の排除や、NodeRAGによる異種ノードへの粒度最適化、StructRAGによるタスク適応型の構造ルーティングといった最新技術のトレンドによって、理論的・実践的に見事に裏付けられている1。  
今後のGraph-Based RAG研究は、医療領域（MedGraphRAG等）や因果推論（CausalKG）で見られるような厳密な階層構造とドメイン知識の統合を基礎としつつ、エージェントベースの動的探索とマルチモーダルな潜在表現学習（S-Path-RAG等）を統合した、汎用的な「適応型推論プラットフォーム」へと進化していくことが予想される。次世代のシステム設計においては、対象とするドメインの知識密度、許容される計算コスト、および要求される推論の深さに応じて、これらのパラダイムを選択的に、かつ柔軟に適用していくことが強く求められる。

#### **引用文献**

> 1. main.tex  
> 2. references.bib  
> 3. Medical Graph RAG: Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation \- ResearchGate, [https://www.researchgate.net/publication/382971086\_Medical\_Graph\_RAG\_Towards\_Safe\_Medical\_Large\_Language\_Model\_via\_Graph\_Retrieval-Augmented\_Generation](https://www.researchgate.net/publication/382971086_Medical_Graph_RAG_Towards_Safe_Medical_Large_Language_Model_via_Graph_Retrieval-Augmented_Generation)  
> 4. arXiv:2307.07697v6 \[cs.CL\] 24 Mar 2024, [https://arxiv.org/pdf/2307.07697](https://arxiv.org/pdf/2307.07697)  
> 5. MemGraphRAG: Memory-based Multi-Agent System for Graph Retrieval-Augmented Generation \- ResearchGate, [https://www.researchgate.net/publication/405684557\_MemGraphRAG\_Memory-based\_Multi-Agent\_System\_for\_Graph\_Retrieval-Augmented\_Generation](https://www.researchgate.net/publication/405684557_MemGraphRAG_Memory-based_Multi-Agent_System_for_Graph_Retrieval-Augmented_Generation)  
> 6. LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora \- arXiv, [https://arxiv.org/html/2510.10114v1](https://arxiv.org/html/2510.10114v1)  
> 7. \[2410.08815\] StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization \- arXiv, [https://arxiv.org/abs/2410.08815](https://arxiv.org/abs/2410.08815)  
> 8. Awesome-GraphRAG: A curated list of resources (surveys, papers, benchmarks, and opensource projects) on graph-based retrieval-augmented generation. \- GitHub, [https://github.com/DEEP-PolyU/Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG)  
> 9. A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models \- arXiv, [https://arxiv.org/html/2501.13958v1](https://arxiv.org/html/2501.13958v1)  
> 10. (PDF) Retrieval-Augmented Generation with Graphs (GraphRAG) \- ResearchGate, [https://www.researchgate.net/publication/387669683\_Retrieval-Augmented\_Generation\_with\_Graphs\_GraphRAG](https://www.researchgate.net/publication/387669683_Retrieval-Augmented_Generation_with_Graphs_GraphRAG)  
> 11. DyG-RAG: Dynamic Graph Retrieval-Augmented Generation with Event-Centric Reasoning, [https://www.alphaxiv.org/abs/2507.13396v1](https://www.alphaxiv.org/abs/2507.13396v1)  
> 12. Graph Retrieval-Augmented Generation: A Survey \- alphaXiv, [https://www.alphaxiv.org/abs/2408.08921](https://www.alphaxiv.org/abs/2408.08921)  
> 13. \[PDF\] LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora, [https://www.semanticscholar.org/paper/LinearRAG%3A-Linear-Graph-Retrieval-Augmented-on-Zhuang-Chen/6586f17b316a65df944e84ddbaf0e3aeab029a75](https://www.semanticscholar.org/paper/LinearRAG%3A-Linear-Graph-Retrieval-Augmented-on-Zhuang-Chen/6586f17b316a65df944e84ddbaf0e3aeab029a75)  
> 14. LINEARRAG: LINEAR GRAPH RETRIEVAL AUG- MENTED GENERATION ON LARGE-SCALE CORPORA \- OpenReview, [https://openreview.net/pdf?id=mCtfkypdm6](https://openreview.net/pdf?id=mCtfkypdm6)  
> 15. \[ICLR 2026\] LinearRAG: Linear Graph Retrieval Augmented Generation on Large-scale Corpora \- GitHub, [https://github.com/DEEP-PolyU/LinearRAG](https://github.com/DEEP-PolyU/LinearRAG)  
> 16. E 2 GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness \- arXiv, [https://arxiv.org/html/2505.24226v4](https://arxiv.org/html/2505.24226v4)  
> 17. E^2GraphRAG: Streamlining Graph-based RAG for High Efficiency and Effectiveness \- ChatPaper, [https://chatpaper.com/chatpaper/de/paper/143836](https://chatpaper.com/chatpaper/de/paper/143836)  
> 18. \[Literature Review\] NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes, [https://www.themoonlight.io/en/review/noderag-structuring-graph-based-rag-with-heterogeneous-nodes](https://www.themoonlight.io/en/review/noderag-structuring-graph-based-rag-with-heterogeneous-nodes)  
> 19. NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes | alphaXiv, [https://www.alphaxiv.org/abs/2504.11544](https://www.alphaxiv.org/abs/2504.11544)  
> 20. NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes \- Hugging Face, [https://huggingface.co/papers/2504.11544](https://huggingface.co/papers/2504.11544)  
> 21. NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes \- arXiv, [https://arxiv.org/html/2504.11544v1](https://arxiv.org/html/2504.11544v1)  
> 22. NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes \- GitHub, [https://github.com/Terry-Xu-666/NodeRAG](https://github.com/Terry-Xu-666/NodeRAG)  
> 23. Gnn-Rag: Graph Neural Retrieval for Large Language Model Reasoning \- arXiv, [https://arxiv.org/html/2405.20139v1](https://arxiv.org/html/2405.20139v1)  
> 24. \[Paper Review\] GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning, [https://nubint.ai/review/gnn-rag-graph-neural-retrieval-for-large-language-model-reasoning](https://nubint.ai/review/gnn-rag-graph-neural-retrieval-for-large-language-model-reasoning)  
> 25. GNN-RAG: Building Powerful Graph Retrieval Systems with PyTorch Geometric \- Medium, [https://medium.com/@ttalati/gnn-rag-building-powerful-graph-retrieval-systems-with-pytorch-geometric-d0972128cb66](https://medium.com/@ttalati/gnn-rag-building-powerful-graph-retrieval-systems-with-pytorch-geometric-d0972128cb66)  
> 26. GNN-RAG: Graph Neural Retrieval for Efficient Large Language Model Reasoning on Knowledge Graphs \- ACL Anthology, [https://aclanthology.org/2025.findings-acl.856.pdf](https://aclanthology.org/2025.findings-acl.856.pdf)  
> 27. GNN-RAG: combining LLMs language abilities with GNNs reasoning in RAG style | by SACHIN KUMAR | Medium, [https://medium.com/@techsachin/gnn-rag-combining-llms-language-abilities-with-gnns-reasoning-in-rag-style-d72200da376c](https://medium.com/@techsachin/gnn-rag-combining-llms-language-abilities-with-gnns-reasoning-in-rag-style-d72200da376c)  
> 28. GNN-RAG: GRAPH NEURAL RETRIEVAL FOR LARGE LANGUAGE MODEL REASONING \- OpenReview, [https://openreview.net/pdf?id=EVuANndPlX](https://openreview.net/pdf?id=EVuANndPlX)  
> 29. Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph Question Answering \- arXiv, [https://arxiv.org/pdf/2603.23512](https://arxiv.org/pdf/2603.23512)  
> 30. S-Path-RAG: Semantic-Aware Shortest-Path Retrieval Augmented Generation for Multi-Hop Knowledge Graph Question Answering \- arXiv, [https://arxiv.org/html/2603.23512v1](https://arxiv.org/html/2603.23512v1)  
> 31. NGM-RAG: Neural Graph Matching based Retrieval-Augmented Generation \- arXiv, [https://arxiv.org/html/2607.11159v1](https://arxiv.org/html/2607.11159v1)  
> 32. Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph | alphaXiv, [https://www.alphaxiv.org/abs/2307.07697](https://www.alphaxiv.org/abs/2307.07697)  
> 33. Deep and Responsible Reasoning of Large Language Model on Knowledge Graph \- arXiv, [https://arxiv.org/html/2307.07697v6](https://arxiv.org/html/2307.07697v6)  
> 34. The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering \- ACL Anthology, [https://aclanthology.org/2025.acl-srw.67.pdf](https://aclanthology.org/2025.acl-srw.67.pdf)  
> 35. Iterative Informed Navigation for Large Language Model Reasoning on Knowledge Graphs \- arXiv, [https://arxiv.org/html/2510.08825v2](https://arxiv.org/html/2510.08825v2)  
> 36. Why Reasoning Fails to Plan: The Fundamental Gap Between Step-by-Step Thinking and Long-Horizon Decision Making in LLM Agents \- CORe Inc., [https://co-r-e.com/method/reasoning-planning-llm-agents](https://co-r-e.com/method/reasoning-planning-llm-agents)  
> 37. StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization | OpenReview, [https://openreview.net/forum?id=GhexuBLxbO](https://openreview.net/forum?id=GhexuBLxbO)  
> 38. StructRAG: Boosting Knowledge Intensive Reasoning of LLMs via Inference-time Hybrid Information Structurization \- Hugging Face, [https://huggingface.co/papers/2410.08815](https://huggingface.co/papers/2410.08815)  
> 39. Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience | alphaXiv, [https://www.alphaxiv.org/abs/2506.00842](https://www.alphaxiv.org/abs/2506.00842)  
> 40. \[論文紹介\#115\]構造化RAG: 推論時のハイブリッド情報構造化によるLLMの知識集約型推論の強化 | hdkworks blog, [https://hdkworks.com/archives/34859](https://hdkworks.com/archives/34859)  
> 41. Medical graphrag: Towards safe medical large language model via, [https://papers.lunadong.com/paper/4814](https://papers.lunadong.com/paper/4814)  
> 42. Towards Safe Medical Large Language Model via Graph Retrieval-Augmented Generation, [https://arxiv.org/html/2408.04187v2](https://arxiv.org/html/2408.04187v2)  
> 43. GraphRAG Goes Medical: Introducing MedGraphRAG \- Gradient Flow, [https://gradientflow.com/graphrag-medgraphrag/](https://gradientflow.com/graphrag-medgraphrag/)  
> 44. What is GraphRAG? \- IBM, [https://www.ibm.com/think/topics/graphrag](https://www.ibm.com/think/topics/graphrag)  
> 45. Knowledge Graph-Based GraphRAG for Clinical Question Answering in Hashimoto's thyroiditis Built from Peer-Reviewed Literature \- DiVA portal, [https://www.diva-portal.org/smash/get/diva2:2084651/FULLTEXT01.pdf](https://www.diva-portal.org/smash/get/diva2:2084651/FULLTEXT01.pdf)  
> 46. Co-MedGraphRAG: A Collaborative Large–Small Model Medical Question-Answering Framework Enhanced by Knowledge Graph Reasoning \- MDPI, [https://www.mdpi.com/2078-2489/17/3/247](https://www.mdpi.com/2078-2489/17/3/247)  
> 47. A Survey of GraphRAG for Customized LLMs: Challenges & Innovations \- Studocu, [https://www.studocu.vn/vn/document/international-university-vnu-hcm/computer-graphics/a-survey-of-graphrag-for-customized-llms-challenges-innovations/159342476](https://www.studocu.vn/vn/document/international-university-vnu-hcm/computer-graphics/a-survey-of-graphrag-for-customized-llms-challenges-innovations/159342476)

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAaCAYAAAC3g3x9AAABZUlEQVR4Xu2UvStGcRTHv/JSQhZ5ySIxGCUpbz3KysBisGM1EJtksBgkC4OXYlEyMD/lj1BkMJmMDAZ8v8699bvHdbtXmfjUp+fpd87z69xzznOBf/4GTfSavge+0vEgp4aeupylIJ7KBCzx0AciKukJXaXVLpZKH32mZ7TKxcQAPaa1PvAdnfSRlml9MvR5yQHtdeeZtNB7+kDbXGyWrtEKd56JqirTJ9oTnHfQc9ocnOVCfVP/Xmh/dKaKNulknFSUfdikNXExQneQc6ppLMMuXKSNsN3rSmQUZAp24Qbs0rlkuDj6d7zRG9jO+fVR1Vuw5S7BchaQ0ZJ4uaUW2aMnUF/v6DRsaEfIGJr2T3u4jvSd66Zj9ApWfbwZ6n0qKn2I1vlAgH68HX1vp7dIvkgKEVekRxf6vMTXXudGFWlgF3Se7tHWREZBhmH907QbXKwwGtIK3fWBnzJIZ2Bv6lEX+z0+AD93OuqrTRVFAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAaCAYAAAC3g3x9AAABSElEQVR4Xu2UvytGYRTHj/woIQORMiAlmSQpUu9gZZBFTBa7gbJJBoOBZGHwoxgo/8Jb/gilDAaZbBgM+Bzn3nqe031f987eT31663xPT/c+59xXpMb/oBPv8DvwE2eCnia8cj3rQZ7JrFjjmQ8S6vESN7HRZZmM4TveYIPLlAm8wGYfVGIAX7CMrXH0e8gpjrp6VbrxEZ+wx2XLuIV1rl4VfaoyvuJwUO/DW+wKarnQe9P7+8DxpKZPtItzaVNRTsQmrRNXpvFQck41iw2xA9ewXWz3BqOOgsyLHbgjduhqHBdHv44vvBfbOb8+KZN4jcfY67KIdLlVXeQsRvBIbIh7uBLHMbp/uofbUnnn9C3e8BxL8sfANJzCFh8EaM+i2M4+41CUFqSED2Jv0ia2Bf1hQ1H0CzrABdzHJal8NbnRv7GO5LeGyA9N5jTC0fgAhQAAAABJRU5ErkJggg==>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAaCAYAAAC3g3x9AAABRUlEQVR4Xu2UvytFYRjHH/kRUQqRMkgGRkkG2UzqGsxWIZtC2SSDnY0ixaL8C7f8ESaDSSkjgwGfr+fees/TcbrnrtenPsN5n6e397zP9xyzf1qDIXzA78RPXEp6uvA29Owl9Vwq5o1XsVCjHW/wADtDLZdZfMc77Ag1MY/X2BMLfzGBL1jFvmzpd5NLnAnrhYzgEz7jaKit4SG2hfVCdKoqvuF0sj6O9zicrDWE7k3394FztTWd6ARX6k1lOTeftCYuFvHUGpxqHvvmG+5gv3n2JjMdJVk13/DYfNONbLk8+jq+8NE8czE+QgFfNo+RHEuLkXq4pYIc0V2e4a75wDS89UxHQPlTDo8sP3N6g1fcxC3zIRaeUCdYwN5YqKGhKZNNTz2iIaU/DyVhKnkujTa4MH/lbfPQD2Q6mkB3O4jdsdCi/ACxzzO2qPrXQAAAAABJRU5ErkJggg==>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABFUlEQVR4Xu3TvUtCURjH8SciKBIiBDEolGrR/gMhEHFwDGpvdbWpQqLFsaHJJaLa+gNCaJKCHPoPcgqi9pAGA+v7cM/15eH4sgQN/uADh/Oce+49L1dkmj9PBFnsIYVZ17+IVdf2Jo0GWrjFAW5wjy3UkO+O7sscymjjEAuDZdnGJ97E8wX6cBXf2DW1MPO4c7Q9kCJ+cIQZU+vPNY5t5ybe0cSaqdlciGf9pxK8vWL6fVmSYLnd6FHV0RHPzJNkBa/4wLqpTZRwAqXtUdGNztjOZTzL+AmiuETMFjRnEuxBwRZc9Fj1Ng67H5LACx4QNzW9jScoyej7IUk84QtX2Mc5HpGTMQ+H0UFJ7Dgb0vsDp/m3+QWDxin9EMHjJAAAAABJRU5ErkJggg==>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABYAAAAaCAYAAACzdqxAAAABTElEQVR4Xu2ULUsEURSGj2BQ0KKCQcNiM4pY/ACDRo27P8Bitqhti0EQwWwUsRgVRAz+CaNgUEwiiBoUP553zr3s3v1gdyyWeeBhhz2HM2feubtmBQWdmMVH/Ane4EjSkbKMX+a9+rzC4aSjgT18wHscb6hFNOAEX/AUe9NyMwN4hAf4htNpOaMH13Ebv3EjLbdmAg+xbP6IK2k5Y8p88BZ+4nxabs2q+SYz+G7N2/RjFcfwHG9xtL6hHVVcwkl8wp2kalYxr2vwneXMVy9MW2ibY/NMRQk3zQdpeO58+8xvch3UtYZpaMlbs+vc+QptqW1jhtpQMQjd+E/5RpSvcl7EXfMXJxSVznjX+erAK46I8tPJ0AAdsUiufOfwDAfrvtMZ1llWPPEFCj1Jx3wX8Nlq/w8fuBZq+tVdWu33v4+voS/2XuBQqBcU/Ae/emZGcKiQxL8AAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,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>

[image7]: <data:image/png;base64,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>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAB0AAAAfCAYAAAAbW8YEAAABbUlEQVR4Xu2VTSsGURSAj1CEYiUhUbIkViJJNhYkpYRfwM7Gx8bK3kcppSQ72SoL/8DWSikkC3bKxkc8p3vvOzO3UTNjFDVPPb3dc+7ctzPn3Eak4I/Tghu4i0NYFk3nj/7hOlZjJ17hbGTHLzCOz9ht11rxOdaUdiRgDD8TuiTm8AGs1IdhG0+xyq5TcYhvOOjFtV+9eItTXq4VL3HCiyeiAS/wBpujqRL7OBpaa6WbOC0ZB0n7o306wQob098eCV7jlt2naGwNR+xaB8o9l5g5MT1bDsW0Yq3ODYi+2noxVS2KGagm7MJVG0+FHv6Ok2IOasM9MZPpM4wfEh0wrToVrp960D3e4aNdh3uYK334ItF+1uExdth1uY3lhuun3kFHI+5IcPdmcCFI/wxt/oHE309HLR5hu5/IiuvntZjq4pgXU3Xq6fyOuH469Kqs4BP2e7lM6FQ+SDDy4clVX0O5MzFflIKCgoKCf8YX62NJ3bd3tOgAAAAASUVORK5CYII=>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAAaCAYAAACHD21cAAABHUlEQVR4Xu3Tr0tDURjG8Vc2wTFRUNa0DIuwpiC2pWGyKChoWxgYljaGsjqGYBAXzLYVk13EOJPBIgwU/AcMRn98H8+9br67SbDtgQ+M9z3n3HPPuTMb50+ZRhElzEa1eeTiAT6TaKKPKg5whxPcYnkwdJA0ztFFdqiuJ/VwY2EnI1nHMwq+QY5w5otxWnjBgm+QOjZ9Mc4FPnGIlOstYc7VfrJrYaK84xplzAwPSopOtG1hUryA3Fvy9keiberYj/FqYXLl14goGqh3mPANsoEPNHxDyaNj4R59VvCGPd9QdMxXmPINso9HLPqGovvTqmuuru3rYLZc/Tv6hC5Rw0P0W1dwiifsWPK7W8bCyoquYxXbFv4ZSVsf59/yBS3hK3hUc+h6AAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAABG0lEQVR4Xu2TsWpCMRSGj+BSBUtBEEfHguAgHQouFgfBzVV3l4LgE9zVxQ51spOTi3QufQKfQF/ATcTFRQfr/5tcTcKl9+oo94MPQk7IyTlJRGLulypcwT/tD3ww4hn4a8TpN0wbaywScAT3cAdf7fCJJpyKnSiQJziGXVGZh6ISmPRgy5kLpAQ/YB4u4BIWjHgSful1oTBjR489Uad7P0dFsqJOzgpCGcCyHhfhBs7go56rwE89/he/X8xOWNIEHmBdz/HUV/XLbDg34WbclLd3U798WB7LZLlvErFfPA178eIGQFvURcxh34kF4vbLJCfqmXDDSP1iCfwaKTeg8eAaPjvzFjW4lctf4xdqWCsUfCb8q6H9iolxOAINPTM0Yd4fNwAAAABJRU5ErkJggg==>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABDElEQVR4XmNgGAWDDzgC8XMg/o+EXwHxLyD+C8QngTgYiJlhGnCBOUD8G4htkMRAmtIYIAaVATEjkhwK4AXiw0B8F4jF0eQkgfghDjk40ATit0C8BohZ0ORMgfgbEF8FYhE0OTjwY4D4PR1dAggaGCByxWjiKGASA6b/WYE4mQHislIoHyvgAeIDDJBQPwZlX2eA2DodiIVhCnEBbP4HhXYlAyT0XaFiOAHM/0Vo4sZA/JUBEr14ATb/g0A0A8TgVjRxFIAv/kEGgwwoRxNHATpA/J4BM/5B7FUMqAZUA7ELTIEtAyR1oad/UHjAACj9gwIRZFAsEM8GYk4keaIAyFu+DJCYIFnzKKAlAAClUjyTTxwUYQAAAABJRU5ErkJggg==>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAaCAYAAACtv5zzAAABaUlEQVR4Xu3UvytFcRjH8UdSRFGklBtJhMU/oCSDRd1iUxaLySCDZFBSZiklAzYzUqa7kRSLhUmJySIMFN6P53x1zvf+OveUDO6nXnU6z7f7nPuc7/mKlPPvU4dBjKMHlcH9WrQG14nSixM8Yw+z2MUx+nCE4Z/VJaQKi3jDPGqiZRnAE+4kwT/QH9/AO8a8mks1DgN6XVKm8YllVHi1cHaw4N8slk7c4wYpr+ZnSxLMf0ns6de8+7lSLzbO2NGtmMGHJHiyOGnBLR7F9no4+i4axdaENYQXkWZMSp4X7xoovQ5HPyjdtqdiW1fHeIGpoN6NfRyITUGnkZUmXEnuBi5ujA/oiJa+MyoFGugY1sXewYhXc9HR6Qjz7f+CDTRtuMalZH+humM2xcaz4tVcijbQdOEMr9jGhNgPnmMGcxhyi73EaqDRcbUjLXaC6sEWZ8/HbpA0v9ZAv4dVsaP9RexI74+sKOfP8wUSjj9xleLAlwAAAABJRU5ErkJggg==>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADsAAAAaCAYAAAAJ1SQgAAADR0lEQVR4Xu2XS6iNURiGX7lECLkl6rhLFKWUIhEnBkoYuA8ZnDIxECZHEiO3mJEoSShyHRjsGLgNSKRc6phQhBIDyeV9ff+y1157rf3/Z3fOPpP91Ns+Z33/ZX3ru6z1A02aNGli9KaGUb1CQxcziOofDjaSvtQ+alVoiDCcaqVWUuNRXhyNj8z+rkULdTH77RSLqI/UH0+fqU3eNXsD+xVqoGcX26mDSEdV44upF9QH6iTsnkvUWWoWdYea7m7IYQF1mRoSGopwgvpNLQkNZAr1gNqIePrMoB5RE0JDhhbmFPWNWg9Ldx899xd1lxoc2FJo8Y5RO0JDHqozTbaDGltp+heNa7B0i+FeKsWiqpW/SX2h5gY2xwjqOXU0NOQwD5YpqUWOotT5RF1HOXJa/W3UfmpANhZDi/MS8YyQ84dhqa/opVDDKaFYvfu4IG0IDbXQxZqQSwml3XFY3cai5SMn31DjQgOsrn7AHJFDKWS7gOL16qPyU83nzfM/Sp+f1HxqKvWYuo9ixa8FKqHamT7UOdgibg1sIZqoohTWchH0/sK17lLhLdUGm6AcVbNa5l2X4jRsG5BzPkrvDlh51BOxoqyAzX1MaIjh6lXdcDdsv1wLi4gcD50IkbNSyBzqO6zxqAH5KIKjYBP0FWaHrtmM+A7gkLPvqYmhIYar150o5/1oWJdTB52ZjaXIc7aEaif0/EPUU9gi6/0lanlmn0Zdhe0CGg/v95GzX2H7dC4qcFevPu2wSei3FilnXcaUkJ6sS3WVkcopRI6UkL5fFE5jV6/qplptH0VUkVWEQ5uPjoi3UX2i0nZ1C7WzQ51cvSG1vxZxVpkZm38VLtViDUb/azxvj1SnjdWl0CFCzsaOdfpfi6Tnp/bJIs7uQuX5oAqt6DtUnnd1Xl2X2dUYHgb2Z9TkzO6jBXuFdMddSL2GPf8IzDH9PqFWUweQvjfPWReQTh8Z60UR0laVio5Q95VDa2BfOpOysTzynNUxUWWmY2PDUJrfQO1jZT3kOav3nodtlw1Dk9E2sTQ01MlQWHrfg/WUM9Tsiivs21cLnPq46FZaYN+5+u1udB5oh30PFz4TdzWqyz1Uv9DQxbRSW9CDjjbpaf4CaQqnNnZZqG0AAAAASUVORK5CYII=>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADkAAAAaCAYAAAANIPQdAAADMklEQVR4Xu2XS6iNURTH/0KRd16Jcj1LFCWKSAkxIKE8p2JABgbKxABl5hGFSAZSHqUQSTlRBpQkugqFREgmKOX1/991lrvPPt/+vnOuewxu51f/zjl7nW/vtddee+39AU2aNOnq9KQGxI0NQGNorP+OBj5GzYwNGQymFlHLqRaqW9A+tPw9D41xBh0I6GzqNfU70Bfqbfn7T+oqNdEfCOhBHaE2x4YATWQ+1Up9oE5S26mLMIenUrepSf5AARuoo+jgih6ivlEzovYJ1AuYk6Mi20KqhHRk+1CnYEFbR3WvNLc5rCDeofpFthS9qcvUithQRF+Ys0+oIZWmNk7DVnVp0KbBrlNbgrYQTfwa9RnpVNZYGlMBroc11F2kg5vJWOodLOq+TxwPwHdqVtA+nXqG7DRTHwdggdFqpfC+610V+fucmhMb8lgGc2hTbCCrYSl1GLYHHf03lWZzYUEpwSaSQrbzyA5UHtoGN6mdsSGPvTCnllAjyhpN7aLeU2tRvZ+UwlKMAnEW6aCFaMUHobrvWtDYKlxx5mXiKfMRlq7Hy1IV1H7ZQ/X3P5fxZxScmJHUS+oT6l+hetiB4kz5S95+bIFV1gfUsKDdJ6mBYrRXvyK7iGnF1I9niyt0VD6owus8zUNj34dlQiF5+1F4ZVUqO7VMsoTqKA+n9lOPYPtc/ZbQ3reOmSuwAGX1HSK7FkB9FpI6H4UfE7+oBUF73iSVokrVEqon6XhKp1ZCgc3qO6TmdC06H71K3kBlZyouF6gTQZvjgdH5OCWyOQqYApc6H2uZpOqBKqwqbS4e9bhKae8oheToY1iljZGDuu71ig2ww1/PXkL1ga3fck6puj6yOUWTlK/yORWkNrRCr9B+V9X+eAO7w+rzByydtsJWJgvtZQUgKwPEPNiBrfvqQdiE9PmQWkntQ7r6Fk1SKX4P5kNDGUM9Rf6tQxmhiayCvXmMK7cVUTRJ3bxaYT40FKXMbthNKD56/pW8Sfq4UmePm4neSm5Rk2NDB5kGu4To3FaWKKUHVvwDGA/b01l1omGoyJxDdYFpBHqH1KWl3gt9p6CX4m1xYwPYSC2OG5t0Zf4AFVinXGlf+O4AAAAASUVORK5CYII=>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA8AAAAbCAYAAACjkdXHAAAA7UlEQVR4Xu2SvQtBURiHXwNRiokUJhOjldFosbLblcgqkzJZDTKYjHab1T9gIGU0WUj8Xue65wP3SinDfeqpc9+Pc+57OkQePyEBOzBqJtzwwS7cwaSRcyUPj5a8/pgQnMAtPMOCnnamAvuwDa+wrKffE4MzmIItEs1VrcKBJqxZaz6Rm3kTV3IkZg1b34/mnl3hwNfNfjiERSXGt8y3PVZiLynBC4mTTOcwKEt1InAKM0acn+cGLkiO8kQD1s0gyeY1jBu5+/vluVYwbeQY/qMliQ14Ixu+mAPJufYwq+QH8KTkeT2CAaXG47+5AQnxL+EKADmBAAAAAElFTkSuQmCC>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAaCAYAAACtv5zzAAABjklEQVR4Xu3UPyhFcRQH8CMUURQlRSQUJikllGQwCptRyiRJkRQlk01KyYBBKauUyUZSLBYykJgskoHC99u5v9fvnnfrvXffYPC+9anbPb/un/M794rk8u9TCn0wCi2QH5wvgZrgOFZa4Qze4QBmYA9OoA2OYSCxOoMUwiJ8wjwUh8vSC2/wJDHegBffhC8YMTWXIjgK8DijTMIPrECeqfnZhQV7MlUa4RnuoNbUbLYlRv+XRZ9+3ZyPSploO9MOR/EUviXGk6WTaniAV9FZ98O9qBBd4yv31nCk+ebE46S4GxCP/fCD4tiei44u23gF40G9HaZFP8BueISeoJZIJdxI9A1cXBtfoME7Pwf3UAUFcCg6BKGwDRuiezBoai5sHVto558P1yl6DfeNRA5KHdzCtSR/oZyYLdH2rJqanw7Rt2HbItMMF/ABOzAmesFLmIJZ6HeLTTi6+9BlCzZ81XoYEv2D8seWauZ58TXRNnIfmsLl7MKbL4k+OQeE+zERWpFlOK7cG99waEUuf55f1AlD4u9V940AAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAaCAYAAABVX2cEAAAA7klEQVR4Xu3TsWrCYBTF8VtwEJ2kLsUncHLo2qXQB+jiIMWXcPE9HEtBEAoOrp2cnH0IM3VQdCh1sbb6P34Iya0SWwpCyYEfJN9JyJdLYpbl/+YWM2xiFmiijBE+Y907+ijq5mN5whfufGFhTV0XF677lhLGiFBJVru0LezqwReHUsUcA+Rcp3Otq9d1qdET9WTtwEc7jSzsXG+Qmg7WuMeVU7cwL12Tmv28lujh0ZnYuefV8oX9Yl76vj5w4wsLa+p+NK/I/uD7urYw+EPzyuMFb6i5LhH9Hq+W/B+naOASQ6xinY6fUdDNWbKckC3WVkHF0VOGHQAAAABJRU5ErkJggg==>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAaCAYAAAC+aNwHAAABK0lEQVR4Xu2TIUsEQRiGP/EMghcOBTnBIlfEZNAkBpFrNpt/wG6yXTFYDCIIFqM/QTAJtrsmiEFBLAqK+gME9XmdHZyd2V0sYtkHHg7m/e7b2dlvzGr+nHHs4jrO4nA+LmYIl7GPp7iReYY3uPBTmjKCu3hnaaGyI3zD+Sj7RgWH+IqLUebRa7zggbmd5tjEj+y3jBYO8AonwqCDD3iNk2EQ4RvcYzsMeviJO+FiATP4aFGDMTw3t/1Vv1iCctVdYNMvqpM66nB0SFXsm9tpL1z0DZL3ipg2NwfPOBcGOk2dalUDfbJtc0/fijJr4Am+41KUeTQXGiANkuYlQZOlgmNLC1bwCfdwNMpyaHRvzd0BP/+6C5fmmiSTV4Rum76Ebt8aTtkv/1jzn3wByukz0ukrHwsAAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAABJklEQVR4Xu3TOy9EQRjG8UdQiBVxKUiWKJCIRiJIbEKrUii01CoRGjQSopKIS0EUxAfQKHQKjVJDp9H6Ev7vmTO775zd+ATnSX7Jzu2dM3POSmXKNKcd6/jAj/OMBbQ1pjanFw/YQldhbACXOEVnYSyLdZ7jC/OFMYvtvIlv7OXtJCvYxQhelBaJi6/RjztMu3F1KDzeVN6uqlHEL47HWsV2/jtLHy7Q4/pikSuliy220Zlra1jh8iquz3bewS+WXL/Fip/4jm6FIwzm7fjY1jeKJyzmY5Y5HLt2lgOFi/SL4+saUlokzk0yjkccKl0cE4ts4F7pcetZw7vC5Fap4RMzxQGfZbxhHxMKF2yv8wavGKvP/Cf2f5jFEW4VPu1Jtfj6yoT8AavDKkEpBZJ/AAAAAElFTkSuQmCC>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAABEUlEQVR4Xu3TvUoDQRSG4SPRQhIQfxq7IFEIKSKI2AhqlyJYWHgRVgpqkVRCIGUiiqiNP2BlZZfO2s47MJUXYeV7MrPLYXY2V7AfPMV+mTmbnWxEihTJpoRjfOMXfxjjAzuYSVdGsoAXXGIeq/66gmXcoI+5ZIONlkO0TWcHaPTuF17mm7RwHnwQDtDojW7RMJ3MYoA1W0p8gOYQp7ZYxD3WxW1KNPGOWtDv41o3JtHyC094NN7wk9PfTXb6lMWd8IotJf8RttELOumKO0ibvAFnkl07eU5drO9CktgAXfccdGmO8Iolfx0OqGOETX8dzZ64A+1gV9yvcIAHfKKarpwS/T9s4UrcqZ9gQyJvXxGXf37pKnkb8UhvAAAAAElFTkSuQmCC>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAZCAYAAAA4/K6pAAAAQElEQVR4XmNgGAWDG/AAcSgQs6BLEAskgXghA8QgssCoAaMGgAD9DBAC4h1A/AgNPwPiX0D8BItcE1jnKBgkAABT3xQI4nTPawAAAABJRU5ErkJggg==>