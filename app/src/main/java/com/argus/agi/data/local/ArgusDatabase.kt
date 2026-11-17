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
    version = 3,
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
        val MIGRATION_2_3: Migration = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE `corpus_sources` ADD COLUMN `licenseIdentifier` TEXT")
                db.execSQL("ALTER TABLE `corpus_sources` ADD COLUMN `licenseName` TEXT")
                db.execSQL("ALTER TABLE `corpus_sources` ADD COLUMN `licenseUrl` TEXT")
                db.execSQL("ALTER TABLE `corpus_sources` ADD COLUMN `attributionText` TEXT")
                db.execSQL("ALTER TABLE `corpus_sources` ADD COLUMN `rightsEvidenceUrl` TEXT")
                db.execSQL("ALTER TABLE `corpus_sources` ADD COLUMN `userDeclarationText` TEXT")
                db.execSQL("ALTER TABLE `corpus_sources` ADD COLUMN `userDeclaredAtEpochMs` INTEGER")
                db.execSQL("ALTER TABLE `corpus_sources` ADD COLUMN `reviewedAtEpochMs` INTEGER")

                // Preserve older explicit user-consent records when the original import
                // marker exists. Statuses without durable evidence are conservatively
                // downgraded; their corpus text and training history remain untouched.
                db.execSQL(
                    """
                    UPDATE `corpus_sources`
                    SET `userDeclarationText` = 'Legacy explicit user approval recorded at import time.',
                        `userDeclaredAtEpochMs` = `importedAtEpochMs`
                    WHERE `rightsStatus` = 'USER_DECLARED'
                      AND (
                        `source` = 'user_approved_conversation'
                        OR instr(`source`, '|rights_declared_by_user=true') > 0
                      )
                    """.trimIndent()
                )
                db.execSQL(
                    "UPDATE `corpus_sources` SET `rightsStatus` = 'UNVERIFIED' " +
                        "WHERE `rightsStatus` = 'USER_DECLARED' AND `userDeclarationText` IS NULL"
                )
                db.execSQL(
                    "UPDATE `corpus_sources` SET `rightsStatus` = 'UNVERIFIED' " +
                        "WHERE `rightsStatus` = 'LICENSE_REVIEWED'"
                )
                // Recompute aggregate document rights from the source records that
                // still carry durable evidence after conservative legacy downgrades.
                db.execSQL(
                    """
                    UPDATE `corpus_documents`
                    SET `rightsStatus` = CASE
                        WHEN EXISTS (
                            SELECT 1 FROM `corpus_sources`
                            WHERE `corpus_sources`.`documentId` = `corpus_documents`.`id`
                              AND `corpus_sources`.`rightsStatus` = 'LICENSE_REVIEWED'
                        ) THEN 'LICENSE_REVIEWED'
                        WHEN EXISTS (
                            SELECT 1 FROM `corpus_sources`
                            WHERE `corpus_sources`.`documentId` = `corpus_documents`.`id`
                              AND `corpus_sources`.`rightsStatus` = 'USER_DECLARED'
                        ) THEN 'USER_DECLARED'
                        ELSE 'UNVERIFIED'
                    END
                    WHERE EXISTS (
                        SELECT 1 FROM `corpus_sources`
                        WHERE `corpus_sources`.`documentId` = `corpus_documents`.`id`
                    )
                       OR `rightsStatus` IN ('USER_DECLARED', 'LICENSE_REVIEWED')
                    """.trimIndent()
                )
            }
        }

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
                INSTANCE ?: Room.databaseBuilder(
                    context.applicationContext,
                    ArgusDatabase::class.java,
                    "argus_ondevice.db"
                )
                    .addMigrations(MIGRATION_1_2, MIGRATION_2_3)
                    .build()
                    .also { database -> INSTANCE = database }
            }
        }
    }
}
