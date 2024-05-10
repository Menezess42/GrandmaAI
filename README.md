<div align="center">
<h1>GrandmAI</h1>
  <img src="https://github.com/menezess42/grandmaai/assets/67249275/4e50e215-1329-49ba-8546-44cdf48401cb" alt="grandmai" width="500">
<h3>A security system powered by AI.</h3>
<p>You can find a video demonstration <b href="https://www.youtube.com/shorts/ZmDJwb9250c">here</b></p>
<hr/>
</div>
<h2>Descrição</h2>
  Projeto que utiliza visão computacional e inteligência artifical para detectar uma ou mais pessoas no video e identificar seu comportamento, caso o comportamento seja identificado como anormal, o sistema alerta.

- <h3>Tecnologias</h3>

  - OpenCV
  - Yolov8Pose
  - Meu própiro <b href="">dataset</b>
  - MLP
  - Celular

<hr/>
<h2>Sobre</h2>
Eu tive essa ideia após ver um vídeo(não consegui achar o vídeo para linkar aqui) na qual uma carcereira estava passando entre celas e foi ataca por um detento. Ela foi salva pelos outros detentos que correram em seu auxilio. No vídeo percebi que teve um espaço de tempo consideravel entre o inicio do ataque e os guardas chegando ao local, com o ataque já prevenido e o detento agressor já contido pelos demais detentos. 
Então pensei, se tivesse um sistema que lendo o feed da camera de segurança, conseguisse detectar o tipo de comportamento e caso seja um comportamento anormal, exemplo o ataque do detento, o sistema emitisse um alerta para os guardas, esse tipo de situação poderia ser evitado.
Mas como obeter dataset de cameras de segurança de presidios seria uma burocracia complicada e demorada, decidi direcionar o projeto para sistemas de segurança residênciais. E mesmo direcionando o projeto para segurança residêncial, não encontrei dataset sobre comportamento, então tive que criar meu próprio que pode ser encontrado no meu perfil no kaggle neste <b href="">link</b>.
<hr/>
<h2>Desenvolvimento</h2>

- <h3>Criando dataset de video</h3>
  Comecei o projeto por criar o dataset de treinamento e teste. Como não tive os fundos necessários para adiquirir algumas cameras de segurança, posicionei meu celuar no muro externo da minha casa.
  Gravei 152 vídeos de comportamentos normais e 152 de anormais, considerei anormais os seguintes comportamentos:

  - Virar em direção a residência e ficar encarando-a
  - Agarrar a grade
  - Tentar subir na grade
  - Portão da garagem:
    - Mexer na fechadura.
    - Tentar erguer.
    - Centar na frente.
  <p>
  Separei a gravação bruta em vídeos de em média 10 segundos, por isso deu 152. Para os comportamentos normais eu simplesmente andei de um ponto a outro da rua, entrando e saindo de frame. Para os comportamentos anormais, eu executei eles cronometrando 10 segundos cada.
  </p>

- <h3>Extraindo dados</h3>
  Após criar o dataset de vídeos separados em pasta Anormais e Normais, decidi criar um dataset numérico, é o trabalho do código  <b href="">make_dataset.py</b>. Esse código é responsável por ler um vídeo frame por frame, passar cada frame pela I.A <b href="">YOLOv8-Pose</b> que identifica os keypoints das pessoas que estão no frame, salvando o resultado de cada frame em um arquivo json dentro de uma pasta. Exemplo, lendo o video anormal_1 ele criara o diretório ./Anormal/anormal_1/frame_x.json na qual x é o numero do frame.

- <h3>Criando e treinando a I.A de detecção</h3>
  O modelo de machine learning é um MLP de 3 camadas escondidas e 1 de saida, sua arquiteura é afunilada, a primeira camada tem 512 neuronios, a segunda tem 256 e a terceira tem 128. Já a camada de saida tem 1 neuronio de resultado binário, se 1 indica Anormal e se 0 indica normal. Tem-se dropout e batchnormalization entre as camadas ocultas, o código de treinamento é o <b href="">ml_training.py</b>.

  O input tem como entrada um vetor de 340 posições. Você pode estar se perguntando, mas se o YOLOv8-Pose retorna 17 pontos chaves para cada pessoa, como que o input é 340?

  Bom o input é 340 pois cada ponto chave é uma coordenada XY. mas ainda sim a conta não baté né ? O input total ser de 340, foi uma escolha para que a MLP conseguisse entender um movimento, com toda sua fluidez. Se eu passase somente um frame por vez, ela iria aprender como se tivesse vendo fotos, para corrigir isso, o código pega os 10 primeiros frames apartir do momento que a pessoa entra em vídeo, são esses 10frames*17XY que cria um imput de 340. Após pegar os 10 primeiros frames, a janela deslizante anda um para frente, ou seja, se pegou de 0 a 9 na primeira interação, na segunda ira pegar de 1 a 10. Escolhi a janela deslizante de tamanho 10 pois 10 frames é poucos segundos de vídeo, o que me permite fazer uma boa leitura do vídeo sem tem um delay perceptivél.
<p align="center"> <img src="https://github.com/Menezess42/GrandmaAI/assets/67249275/43681e1d-ba2d-4ca9-adf3-0eada612ab35" alt="GrandmAI model" width="500"> </p>
<hr/>
<h2>GrandmAI no mundo real</h2>
Bom, o tópico anterior fala sobre o desenvolvimento, mais simples, os vídeos de treinamento e test contém somente um ator(eu) executando os movimentos, como é a GrandmAI na prática ?
Os testes e coleta de dados sobre a execução no mundo real ainda estão sendo colhidos. Em termos de execução, o que difere do código de treinamento para o código "final" é o fato de que usamos o track do YOLOv8 para rastrear cada indivíduo em video, pois num cenário real não é só uma pessoa de cada vez que ira entrar e sair de frame. Com esse track salvamos em um dicionário os frames de cada pessoa rastreada, sendo o id de rastreio a chave e o valor sendo um vetor de frames, quando o vetor atinge len=10, um modelo da IA é instanciado e o vetor é passado por essa instancia. Após isso, a primeira posição do veotr é descartada e a próxima leitura do yolov8 apra quela pessoa é esperada.
