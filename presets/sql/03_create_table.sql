-- 用户相关
CREATE TYPE userrole AS ENUM ('VISITOR', 'MEMBER', 'ADMIN');
CREATE TABLE IF NOT EXISTS users
(
    id              SERIAL,
    username        VARCHAR  NOT NULL,
    email           VARCHAR  NOT NULL,
    role            userrole NOT NULL,
    hashed_password VARCHAR  NOT NULL,
    is_active       BOOLEAN  NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (email)
);

CREATE TABLE IF NOT EXISTS chat_sessions
(
    uid            uuid      NOT NULL,
    owner_id       INTEGER   NOT NULL,
    description    VARCHAR   NOT NULL,
    checkpoint_uid uuid,
    interrupted    BOOLEAN   NOT NULL,
    create_time    TIMESTAMP NOT NULL,
    update_time    TIMESTAMP NOT NULL,
    PRIMARY KEY (uid),
    FOREIGN KEY (owner_id) REFERENCES users
);


-- 项目相关
CREATE TABLE IF NOT EXISTS projects
(
    id          SERIAL,
    owner_id    INTEGER   NOT NULL,
    name        VARCHAR   NOT NULL,
    description VARCHAR   NOT NULL,
    public      BOOLEAN   NOT NULL,
    create_time TIMESTAMP NOT NULL,
    update_time TIMESTAMP NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (owner_id) REFERENCES users
);

CREATE TABLE IF NOT EXISTS project_tags
(
    id    SERIAL,
    name  VARCHAR NOT NULL,
    color VARCHAR NOT NULL,
    PRIMARY KEY (id)
);

CREATE TABLE IF NOT EXISTS project_tag_link
(
    project_id INTEGER NOT NULL,
    tag_id     INTEGER NOT NULL,
    PRIMARY KEY (project_id, tag_id),
    FOREIGN KEY (project_id) REFERENCES projects,
    FOREIGN KEY (tag_id) REFERENCES project_tags
);

CREATE TABLE IF NOT EXISTS project_star
(
    project_id INTEGER NOT NULL,
    user_id    INTEGER NOT NULL,
    PRIMARY KEY (project_id),
    FOREIGN KEY (project_id) REFERENCES projects,
    FOREIGN KEY (user_id) REFERENCES users
);

-- 资源相关
CREATE TABLE IF NOT EXISTS bio_dbs
(
    id          SERIAL,
    name        VARCHAR NOT NULL,
    description VARCHAR,
    path        VARCHAR NOT NULL,
    size        VARCHAR NOT NULL,
    last_update VARCHAR,
    PRIMARY KEY (id),
    UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS reference_genomes
(
    id                  SERIAL,
    ncbi_tax_id         INTEGER NOT NULL,
    name                VARCHAR NOT NULL,
    accession           VARCHAR NOT NULL,
    aliases             VARCHAR,
    genome_sequences    VARCHAR NOT NULL,
    annotation_features VARCHAR NOT NULL,
    bowtie2_index       VARCHAR,
    bwa_index           VARCHAR,
    hisat2_index        VARCHAR,
    star_index          VARCHAR,
    minimap2_index      VARCHAR,
    PRIMARY KEY (id),
    UNIQUE (name),
    UNIQUE (ncbi_tax_id)
);

CREATE TABLE IF NOT EXISTS samples
(
    uid         uuid      NOT NULL,
    owner_id    INTEGER   NOT NULL,
    project_id  INTEGER   NOT NULL,
    sample_name VARCHAR   NOT NULL,
    meta_data   jsonb,
    create_time TIMESTAMP NOT NULL,
    PRIMARY KEY (uid),
    FOREIGN KEY (owner_id) REFERENCES users,
    FOREIGN KEY (project_id) REFERENCES projects
);

CREATE TYPE samplefiletype AS ENUM ('SEQUENCING_R1','SEQUENCING_R2','SPECTRUM','IMAGE');
CREATE TABLE IF NOT EXISTS sample_files
(
    uid           uuid           NOT NULL,
    sample_uid    uuid,
    data_type     samplefiletype NOT NULL,
    file_path     VARCHAR        NOT NULL,
    file_format   VARCHAR        NOT NULL,
    file_size     INTEGER        NOT NULL,
    md5_checksum  VARCHAR        NOT NULL,
    uploaded_time TIMESTAMP      NOT NULL,
    PRIMARY KEY (uid),
    FOREIGN KEY (sample_uid) REFERENCES samples
);

-- 工具相关
CREATE TABLE IF NOT EXISTS images
(
    uid         uuid    NOT NULL,
    name        VARCHAR NOT NULL,
    version     VARCHAR,
    description VARCHAR,
    homepage    VARCHAR,
    paper_link  VARCHAR,
    image       jsonb,
    PRIMARY KEY (uid)
);

CREATE TABLE IF NOT EXISTS tool_docs
(
    uid          uuid    NOT NULL,
    help_command VARCHAR NOT NULL,
    description  VARCHAR,
    content      VARCHAR NOT NULL,
    image_uid    uuid,
    PRIMARY KEY (uid),
    FOREIGN KEY (image_uid) REFERENCES images
);

CREATE TABLE IF NOT EXISTS tool_tags
(
    id   SERIAL,
    name VARCHAR NOT NULL,
    PRIMARY KEY (id),
    UNIQUE (name)
);

CREATE TABLE IF NOT EXISTS tool_groups
(
    id        SERIAL,
    name      VARCHAR NOT NULL,
    parent_id INTEGER,
    tag_id    INTEGER,
    PRIMARY KEY (id),
    FOREIGN KEY (parent_id) REFERENCES tool_groups,
    FOREIGN KEY (tag_id) REFERENCES tool_tags
);

CREATE TYPE tooltype AS ENUM ('DOCKER', 'COMMAND_LINE');
CREATE TABLE IF NOT EXISTS tools
(
    uid              uuid         NOT NULL,
    name             VARCHAR      NOT NULL,
    image_uid        uuid,
    description      VARCHAR      NOT NULL,
    des_vector       vector(2560) NOT NULL,
    tool_type        tooltype     NOT NULL,
    group_id         INTEGER,
    help_doc_uid     uuid,
    command_template VARCHAR      NOT NULL,
    dynamic_params   jsonb,
    static_params    VARCHAR      NOT NULL,
    file_mounts      jsonb,
    mkdir_output     BOOLEAN      NOT NULL,
    use_temp_dir     BOOLEAN      NOT NULL,
    PRIMARY KEY (uid),
    FOREIGN KEY (image_uid) REFERENCES images,
    FOREIGN KEY (group_id) REFERENCES tool_groups,
    FOREIGN KEY (help_doc_uid) REFERENCES tool_docs
);

CREATE TABLE IF NOT EXISTS tool_tag_link
(
    tool_uid uuid    NOT NULL,
    tag_id   INTEGER NOT NULL,
    PRIMARY KEY (tool_uid, tag_id),
    FOREIGN KEY (tool_uid) REFERENCES tools,
    FOREIGN KEY (tag_id) REFERENCES tool_tags
);

-- 工作流相关
CREATE TYPE workflowtype AS ENUM ('SUBMODULE','TEMPLATE');
CREATE TABLE IF NOT EXISTS workflows
(
    uid      uuid         NOT NULL,
    name     VARCHAR      NOT NULL,
    owner_id INTEGER      NOT NULL,
    wf_type  workflowtype NOT NULL,
    workflow jsonb,
    public   BOOLEAN      NOT NULL,
    PRIMARY KEY (uid),
    FOREIGN KEY (owner_id) REFERENCES users
);

CREATE TYPE status AS ENUM ('WAITING','RUNNING','ERROR','SUCCESS');
CREATE TABLE IF NOT EXISTS run_instances
(
    uid         uuid    NOT NULL,
    name        VARCHAR NOT NULL,
    project_id  INTEGER,
    owner_id    INTEGER NOT NULL,
    workflow    jsonb,
    status      status  NOT NULL,
    create_time TIMESTAMP,
    start_time  TIMESTAMP,
    end_time    TIMESTAMP,
    PRIMARY KEY (uid),
    FOREIGN KEY (project_id) REFERENCES projects,
    FOREIGN KEY (owner_id) REFERENCES users
);

CREATE TABLE IF NOT EXISTS task_instances
(
    uid          uuid    NOT NULL,
    instance_uid uuid,
    tool_uid     uuid    NOT NULL,
    owner_id     INTEGER NOT NULL,
    name         VARCHAR NOT NULL,
    commands     VARCHAR,
    tool_output  jsonb,
    status       status  NOT NULL,
    system       VARCHAR,
    hostname     VARCHAR,
    create_time  TIMESTAMP,
    start_time   TIMESTAMP,
    end_time     TIMESTAMP,
    PRIMARY KEY (uid),
    FOREIGN KEY (instance_uid) REFERENCES run_instances,
    FOREIGN KEY (tool_uid) REFERENCES tools,
    FOREIGN KEY (owner_id) REFERENCES users
);

CREATE TABLE IF NOT EXISTS task_monitor_logs
(
    id        SERIAL,
    task_uid  uuid,
    cpu_usage DOUBLE PRECISION NOT NULL,
    mem_usage DOUBLE PRECISION NOT NULL,
    mem_used  DOUBLE PRECISION NOT NULL,
    io_in     DOUBLE PRECISION NOT NULL,
    io_out    DOUBLE PRECISION NOT NULL,
    time      TIMESTAMP        NOT NULL,
    PRIMARY KEY (id),
    FOREIGN KEY (task_uid) REFERENCES task_instances
);