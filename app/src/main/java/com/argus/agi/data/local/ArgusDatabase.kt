package com.argus.agi.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.argus.agi.data.local.dao.ChatMessageDao
import com.argus.agi.data.local.dao.CorpusDao
import com.argus.agi.data.local.dao.TrainingQueueDao
import com.argus.agi.data.local.dao.WorldMemoryDao
import com.argus.agi.data.local.entity.ChatMessageEntity
import com.argus.agi.data.local.entity.CorpusChunkEntity
import com.argus.agi.data.local.entity.CorpusDocumentEntity
import com.argus.agi.data.local.entity.CorpusSourceEntity
import com.argus.agi.data.local.entity.TrainingItemEntity
import com.argus.agi.data.local.entity.WorldMemoryEntity

@Database(
    entities = [
        ChatMessageEntity::class,
        WorldMemoryEntity::class,
        TrainingItemEntity::class,
        CorpusDocumentEntity::class,
        CorpusSourceEntity::class,
        CorpusChunkEntity::class
    ],
    version = 2,
    exportSchema = false
)
abstract class ArgusDatabase : RoomDatabase() {
    abstract fun chatMessageDao(): ChatMessageDao
    abstract fun worldMemoryDao(): WorldMemoryDao
    abstract fun trainingQueueDao(): TrainingQueueDao
    abstract fun corpusDao(): CorpusDao

    companion object {
        /**
         * Version 1 already contains user chat, memory, and training queue data.
         * Version 2 adds retained corpus tables and a nullable document reference.
         * This migration is additive; it does not erase any existing user data.
         */
        val MIGRATION_1_2: Migration = object : Migration(1, 2) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL(
                    "ALTER TABLE `training_items` ADD COLUMN `documentId` TEXT"
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_training_items_documentId` " +
                        "ON `training_items` (`documentId`)"
                )

                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `corpus_documents` (
                        `id` TEXT NOT NULL,
                        `contentFingerprint` TEXT NOT NULL,
                        `fingerprintAlgorithm` TEXT NOT NULL,
                        `source` TEXT NOT NULL,
                        `title` TEXT NOT NULL,
                        `language` TEXT,
                        `canonicalUrl` TEXT,
                        `characterCount` INTEGER NOT NULL,
                        `chunkCount` INTEGER NOT NULL,
                        `datasetSplit` TEXT NOT NULL,
                        `validationText` TEXT,
                        `plannedEpochs` INTEGER NOT NULL,
                        `rightsStatus` TEXT NOT NULL,
                        `status` TEXT NOT NULL,
                        `createdAtEpochMs` INTEGER NOT NULL,
                        `lastTrainedAtEpochMs` INTEGER,
                        `completedPassCount` INTEGER NOT NULL,
                        PRIMARY KEY(`id`)
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS " +
                        "`index_corpus_documents_contentFingerprint` " +
                        "ON `corpus_documents` (`contentFingerprint`)"
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_corpus_documents_createdAtEpochMs` " +
                        "ON `corpus_documents` (`createdAtEpochMs`)"
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_corpus_documents_datasetSplit_createdAtEpochMs` " +
                        "ON `corpus_documents` (`datasetSplit`, `createdAtEpochMs`)"
                )

                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `corpus_sources` (
                        `id` TEXT NOT NULL,
                        `documentId` TEXT NOT NULL,
                        `source` TEXT NOT NULL,
                        `canonicalUrl` TEXT,
                        `language` TEXT,
                        `rightsStatus` TEXT NOT NULL,
                        `importedAtEpochMs` INTEGER NOT NULL,
                        PRIMARY KEY(`id`),
                        FOREIGN KEY(`documentId`) REFERENCES `corpus_documents`(`id`)
                            ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_corpus_sources_documentId` " +
                        "ON `corpus_sources` (`documentId`)"
                )

                db.execSQL(
                    """
                    CREATE TABLE IF NOT EXISTS `corpus_chunks` (
                        `id` TEXT NOT NULL,
                        `documentId` TEXT NOT NULL,
                        `chunkIndex` INTEGER NOT NULL,
                        `content` TEXT NOT NULL,
                        `characterCount` INTEGER NOT NULL,
                        PRIMARY KEY(`id`),
                        FOREIGN KEY(`documentId`) REFERENCES `corpus_documents`(`id`) 
                            ON UPDATE NO ACTION ON DELETE CASCADE
                    )
                    """.trimIndent()
                )
                db.execSQL(
                    "CREATE INDEX IF NOT EXISTS `index_corpus_chunks_documentId` " +
                        "ON `corpus_chunks` (`documentId`)"
                )
                db.execSQL(
                    "CREATE UNIQUE INDEX IF NOT EXISTS " +
                        "`index_corpus_chunks_documentId_chunkIndex` " +
                        "ON `corpus_chunks` (`documentId`, `chunkIndex`)"
                )
            }
        }

        @Volatile
        private var INSTANCE: ArgusDatabase? = null

        fun getInstance(context: Context): ArgusDatabase {
            return INSTANCE ?: synchronized(this) {
                val instance = Room.databaseBuilder(
                    context.applicationContext,
                    ArgusDatabase::class.java,
                    "argus_ondevice.db"
                )
                    .addMigrations(MIGRATION_1_2)
                    .build()
                INSTANCE = instance
                instance
            }
        }
    }
}
