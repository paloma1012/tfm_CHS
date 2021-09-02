# Máster en Ciencia de Datos e Ingeniería de Datos en la Nube
## Módulo 14
### Daniel González, Javier Cózar, Jesús Martínez, Juan Ignacio Alonso

## Capstone XIV
---

En este capstone pondremos a prueba todo lo aprendido con respecto a la creación de aplicaciones visuales (_Dash_) y cómo crear un proceso automatizado para su puesta en producción y despliegue de nuevos cambios (_CI/CD con Code Pipeline_).

## Descripción

En este capstone partimos del problema planteado en el módulo IX donde desarrollamos una aplicación visual con _Dash_ en la que visualizabamos datos de accesos a nuestra aplicación web a través de varias redes sociales (_facebook_, _twitter_ e _instagram_).

Hemos detectado que mucha gente está accediendo a través de una nueva red social, _twitch_, por lo que hemos decidimos ampliar el dataset (fichero `social_network.parquet`) para que incluya los datos de esta nueva red social e incluya los datos en nuestra aplicación de Dash.

El código proporcionado en la carpeta `05-codepipeline-dash` incluye el código de la aplicación de Dash (`app.py`) actualizado para trabajar con los nuevos datos. El fichero `social_network.parquet` se encuentra en la ruta `s3://mc1-m14-capstone-data/social_network.parquet`.

El objetivo de este capstone es el de crear un proyecto de CI/CD que, cada vez que subimos un nuevo commit a un repositorio en GitHub, despliegua la aplicación de Dash en una instancia de EC2 usando el servicio `CodeDeploy`, creando la revisión que éste necesita a través del servicio `CodeBuild`, ambos servicios orquestados mediante `CodePipeline`. Podemos ver la interacción entre los diferentes servicios en el diagrama a continuación:

![diagram](images/diagram.png)


## Entregable

Se deberá realizar el capstone en el classroom del módulo 14. Asimismo se deberá subir este fichero `README.md` a la tarea de **campus virtual** completando en el primer paso de la sección `Pasos a seguir` la información requerida (url del repositorio de github público creado, el cual **deberá permanecer accesible al menos hasta su corrección**). Opcionalmente se podrá completar también la información solicitada en el último paso de la sección `Pasos a seguir` (se valorará positivamente).


## Pasos a seguir

1. Crear un repositorio **público** en GitHub. **<<INDICAR AQUÍ LA URL A DICHO REPOSITORIO>>**
    - El repositorio deberá inicializarse con el código proporcionado dentro de la carpeta `05-codepipeline-dash` (los siguientes comandos de git pueden ser necesarios `git init`, `git add`, `git commit`, `git push`)
2. Crear un rol de IAM para el servicio CodeBuild. Asignarle la policy `AmazonS3ReadOnlyAccess`
3. Crear un rol de IAM para el servivio CodeDeploy (CodeDeploy).
4. Crear un rol de IAM para el servivio EC2.  Asignarle la policy `AmazonS3ReadOnlyAccess`
5. Crear una instancia de EC2
    - Amazon Linux 2 AMI 64-bit (x86)
    - Tipo t2.micro
    - Especificar el rol creado en el paso 4
    - Añadir una `Tag` con Key `mc1-m14-capstone`
    - Especificar la opción `Create a new security group`
        - Añadir una entrada para SSH desde cualquier source
        - Añadir una entrada para HTTP desde cualquier source
    - Al lanzar la instancia, indicar la opción de crear una nueva `Key Pair` y vincularla a la instancia
6. Crear una aplicación de CodeDeploy
    - Compute platform: EC2/On-premises
7. Crear un Deployment Group
    - Seleccionar el rol de IAM creado en el paso 3
    - Deployment type: `In-place`
    - Environment configuration `Amazon EC2 instances`
        - Especificar una `Tag` con Key `mc1-m14-capstone`
        - En la sección Agent configuration with AWS Systems Manaher seleccionar `Never` para que **no** instale AWS CodeDeploy Agent
    - Deployment settings: `CodeDeployDefault.AllAtOnce`
    - **Desactivar** la casilla de `Enable load balancing`
8. Conectarse a la instancia de EC2 por SSH
9. Instalar el agente de codedeploy. Para ello ejecutar en la terminal, una vez conectados por SSH, los siguientes comandos:
    - sudo yum update
    - sudo yum install ruby wget
    - wget https://aws-codedeploy-us-east-1.s3.us-east-1.amazonaws.com/latest/install
    - chmod +x ./install
    - sudo ./install auto
    - sudo service codedeploy-agent status
10. Crear un nuevo pipeline en CodePipeline
    - Especificar la opción de `New Service role` en el paso 1
    - Como Source provider elegir `GitHub  (Version 1)`
        - Conectar con tu cuenta de GitHub a través del botón adyacente para permitir a AWS el acceso, y elegir el repositorio **público** creado en el paso 1 y la branch `master`.
        - Dejar la opción de `GitHib webhooks`
    - Como Build provider elegir `AWS CodeBuild`
        - Usar el botón de Create proyect para crear un nuevo proyecto de CodeBuild
            - Environment image: `Managed image`
            - Operating system: `Amazon Linux 2`
            - Runtime(s): `Standard`
            - Image: `aws/codebuild/amazonlinux2-x86_64-standard:3.0`
            - Image version: `Always use the latest image for this runtime version`
            - Environment type: `Linux`
            -  Elegir el rol de IAM creado en el paso 2
            - Buildspec: `Use a buildspec file`
        - Como build type seleccionar `Single build`
    - Como Deploy provider seleccionar `AWS CodeDeploy`
        - Seleccionar la aplicación y el deployment group creados en los pasos 6 y 7 respectivamente
    - Crear el pipeline
11. Verificar que el pipeline se completa correctamente
12. Verificar que podemos acceder a la aplicación de _Dash_ desplegada en la instancia de EC2 usando su `DNS`
13. Efectuar un cambio en la aplicación. Para ello, modificaremos el código del repositorio cambiando el título de la aplicación (primer elemento `H1`) de tal manera que sea `'<TU NOMBRE> - Dashboard Social Networks'`. Desplegar el cambio (`git add`, `git commit` y `git push`) y verificar que la aplicación se actualiza finalizado el pipeline activado tras el `git push`.
14. **Opcional**: ¿Se te ocurre algún KPI que mostrar sobre la nueva red social twitch? Modifica la aplicación de Dash y despliega el cambio de forma similar a como se hizo en el paso 13. **Comenta y justifica a continuación** el cambio realizado.
