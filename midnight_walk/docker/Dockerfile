FROM python:3.11-slim

ARG SERVER_WHEEL_FILE
ARG ALEMBIC_TAR_FILE

ENV APP_HOME=/opt/midnight \
    MIDNIGHT_SERVER_HOST=0.0.0.0

WORKDIR ${APP_HOME}

COPY ${SERVER_WHEEL_FILE} /tmp/${SERVER_WHEEL_FILE}
RUN pip install --no-cache-dir /tmp/${SERVER_WHEEL_FILE} && rm /tmp/${SERVER_WHEEL_FILE}

COPY ${ALEMBIC_TAR_FILE} /tmp/alembic_bundle.tar
RUN mkdir -p ${APP_HOME} && tar -xf /tmp/alembic_bundle.tar -C ${APP_HOME} && rm /tmp/alembic_bundle.tar

COPY entrypoint.sh ${APP_HOME}/entrypoint.sh
RUN chmod +x ${APP_HOME}/entrypoint.sh

EXPOSE 9000

CMD ["bash", "/opt/midnight/entrypoint.sh"]
